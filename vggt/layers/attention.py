# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings
import time
import matplotlib.pyplot as plt

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from magi_attention.functional.flex_flash_attn import flex_flash_attn_func
from magi_attention.common.enum import AttnMaskType
# from xformers.ops import memory_efficient_attention

XFORMERS_AVAILABLE = False

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, attn_bias=None, pos=None, motion_scores=None, save_attn=False, kv_cache=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache["k"], kv_cache["v"]
            if k_cache is not None and v_cache is not None:
                k = torch.cat([k_cache, k], dim=2)
                v = torch.cat([v_cache, v], dim=2)
            kv_cache["k"] = k
            kv_cache["v"] = v
        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            # if motion_scores is not None:
            #     motion_scores_expanded = motion_scores.unsqueeze(1)
            #     attn[:, :, 0, :] = attn[:, :, 0, :] + motion_scores_expanded
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            # save_attention_heatmap(attn)
            x = attn @ v
        if save_attn:
            q = q * self.scale
            # save_attention_heatmap(attn)
            attn_weight = q[:, :, ::save_attn, :] @ k.transpose(-2, -1)
            attn_weight = attn_weight.softmax(dim=-1)
            attn_weight = self.attn_drop(attn_weight)

        extra_tokens = 5
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # if save_attn:
        #     attn_weight_list=[]
        #     for i in range(attn_weight.shape[-1] // save_attn)
        #         attn_weight_list.append(attn_weight[:, :, :, i * save_attn + extra_tokens, (i + 1) * save_attn])
        #     return x, torch.stack(attn_weight_list, dim=2)
        if save_attn:
            return x, attn_weight
        elif kv_cache is not None:
            return x, kv_cache
        else:
            return x, None

class SparseAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None, motion_scores=None, save_attn=False, kv_cache=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        # v  = v.to(dtype=q.dtype)
        q = q.to(dtype=v.dtype)
        k = k.to(dtype=v.dtype)

        def flex_flash_attention(
            q: Tensor,              # [B, L, Hq, D]
            k: Tensor,              # [B, L, Hkv, D]
            v: Tensor,              # [B, L, Hkv, D]
            block_size: int,              # N：块大小
            attn_mode: str = "full",      # "full" | "causal" | "inv_causal" | "bi_causal"
            softmax_scale: float | None = None,
            dropout_p: float = 0.0,       
            disable_fwd_atomic_reduction: bool = True,
            return_lse: bool = False,
        ):
            """
            将 batch 的单序列“前缀分块”掩码展开为 FFA 的 ranges 并计算注意力。
            形状约定：q/k/v = [B, L, H, D]；输出与 q 同形状。
            """
            assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "q/k/v 必须是 [B, L, H, D]"
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            B, L, Hq, D = q.shape
            Bk, Lk, Hkv, Dk = k.shape
            assert (Bk, Lk, Dk) == (B, L, D), "k/v 的 B/L/D 必须与 q 匹配"
            assert v.shape == k.shape, "k/v 形状需要一致"
            assert block_size > 0, "block_size 必须 > 0"

            # 展平到 (B*L, H, D)
            q_flat = q.reshape(B * L, Hq, D).contiguous()
            k_flat = k.reshape(B * L, Hkv, D).contiguous()
            v_flat = v.reshape(B * L, Hkv, D).contiguous()

            # 构造掩码 ranges：对每个样本复制单序列规则并按 base 偏移
            # blocks_per_sample = (L + block_size - 1) // block_size
            blocks_per_sample = L // block_size
            q_ranges = []
            k_ranges = []
            attn_types = []
            for b in range(B):
                base = b * L
                for i in range(blocks_per_sample):
                    q0 = base + i * block_size
                    q1 = min(base + (i + 1) * block_size, base + L)
                    k0 = base
                    k1 = min(base + (i + 1) * block_size, base + L)
                    q_ranges.append([q0, q1])
                    k_ranges.append([k0, k1])
                    attn_types.append(0)
                if (L % block_size != 0):
                    q_ranges.append([base + blocks_per_sample * block_size, base + L])
                    k_ranges.append([base, base + blocks_per_sample * block_size])
                    attn_types.append(0)
                # q_ranges.append([base, base + L])
                # k_ranges.append([base, base + L])
                # attn_types.append(0)

            q_ranges_t = torch.tensor(q_ranges, dtype=torch.int32, device=q.device)
            k_ranges_t = torch.tensor(k_ranges, dtype=torch.int32, device=q.device)
            attn_types_t = torch.tensor(attn_types, dtype=torch.int32, device=q.device)

            out_flat, lse = flex_flash_attn_func(
                q_flat, k_flat, v_flat,
                q_ranges=q_ranges_t,
                k_ranges=k_ranges_t,
                attn_type_map=attn_types_t,
                softmax_scale=softmax_scale,  # None = 1/sqrt(D)
                disable_fwd_atomic_reduction=disable_fwd_atomic_reduction,  # q_ranges 不重叠时可 True
            )

            out = out_flat.view(B, L, Hq, D)
            return out

        x = flex_flash_attention(
            q, k, v,
            block_size=attn_bias,
            attn_mode="causal",
            dropout_p=self.attn_drop.p if self.training else 0.0
        )
        # x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, is_causal=True)
        # print(x.shape, x_new.shape)
        # breakpoint()
        # diff = (x - x_new).float()
        # print("max |diff|:", diff.abs().max().item())
        # # print("mean |diff|:", diff.abs().mean().item())
        x = x.reshape(B, N, C)
        # x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, None

class CausalAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None, motion_scores=None, save_attn=False) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        # v  = v.to(dtype=q.dtype)
        q = q.to(dtype=v.dtype)
        k = k.to(dtype=v.dtype)

        # x = flex_flash_attention(
        #     q, k, v,
        #     block_size=attn_bias,
        #     attn_mode="causal",
        #     dropout_p=self.attn_drop.p if self.training else 0.0
        # )

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, is_causal=True)
        # x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        # x = spas_sage2_attn_meansim_cuda(q, k, v, simthreshd1=0.6, cdfthreshd=0.97, pvthreshd=15, is_causal=False)
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, None

class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None, motion_scores=None, save_attn=False, kv_cache=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, None

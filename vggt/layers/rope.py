# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


# Implementation of 2D Rotary Position Embeddings (RoPE).

# This module provides a clean implementation of 2D Rotary Position Embeddings,
# which extends the original RoPE concept to handle 2D spatial positions.

# Inspired by:
#         https://github.com/meta-llama/codellama/blob/main/llama/model.py
#         https://github.com/naver-ai/rope-vit


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class PositionGetter:
    """Generates and caches 2D spatial positions for patches in a grid.

    This class efficiently manages the generation of spatial coordinates for patches
    in a 2D grid, caching results to avoid redundant computations.

    Attributes:
        position_cache: Dictionary storing precomputed position tensors for different
            grid dimensions.
    """

    def __init__(self):
        """Initializes the position generator with an empty cache."""
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Generates spatial positions for a batch of patches.

        Args:
            batch_size: Number of samples in the batch.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.

        Returns:
            Tensor of shape (batch_size, height*width, 2) containing y,x coordinates
            for each position in the grid, repeated for each batch item.
        """
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding implementation.

    This module applies rotary position embeddings to input tokens based on their
    2D spatial positions. It handles the position-dependent rotation of features
    separately for vertical and horizontal dimensions.

    Args:
        frequency: Base frequency for the position embeddings. Default: 100.0
        scaling_factor: Scaling factor for frequency computation. Default: 1.0

    Attributes:
        base_frequency: Base frequency for computing position embeddings.
        scaling_factor: Factor to scale the computed frequencies.
        frequency_cache: Cache for storing precomputed frequency components.
    """

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        """Initializes the 2D RoPE module."""
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes frequency components for rotary embeddings.

        Args:
            dim: Feature dimension (must be even).
            seq_len: Maximum sequence length.
            device: Target device for computations.
            dtype: Data type for the computed tensors.

        Returns:
            Tuple of (cosine, sine) tensors for frequency components.
        """
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            # Compute frequency bands
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)

            # Generate position-dependent frequencies
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)

            # Compute and cache frequency components
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """Performs feature rotation by splitting and recombining feature dimensions.

        Args:
            x: Input tensor to rotate.

        Returns:
            Rotated feature tensor.
        """
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor
    ) -> torch.Tensor:
        """Applies 1D rotary position embeddings along one dimension.

        Args:
            tokens: Input token features.
            positions: Position indices.
            cos_comp: Cosine components for rotation.
            sin_comp: Sine components for rotation.

        Returns:
            Tokens with applied rotary position embeddings.
        """
        # Embed positions with frequency components
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]

        # Apply rotation
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Applies 2D rotary position embeddings to input tokens.

        Args:
            tokens: Input tensor of shape (batch_size, n_heads, n_tokens, dim).
                   The feature dimension (dim) must be divisible by 4.
            positions: Position tensor of shape (batch_size, n_tokens, 2) containing
                      the y and x coordinates for each token.

        Returns:
            Tensor of same shape as input with applied 2D rotary position embeddings.

        Raises:
            AssertionError: If input dimensions are invalid or positions are malformed.
        """
        # Validate inputs
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions must have shape (batch_size, n_tokens, 2)"

        # Compute feature dimension for each spatial direction
        feature_dim = tokens.size(-1) // 2

        # Get frequency components
        max_position = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(feature_dim, max_position, tokens.device, tokens.dtype)

        # Split features for vertical and horizontal processing
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)

        # Apply RoPE separately for each dimension
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_comp, sin_comp)

        # Combine processed features
        return torch.cat((vertical_features, horizontal_features), dim=-1)


class PositionGetter3D:
    """
    cache (t, y, x) coordinate
    usage:
        pos = position_getter_3d(B, T, H, W, device)   # (B, T*H*W, 3)
    """
    def __init__(self):
        self.cache: Dict[Tuple[int, int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, T: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (T, H, W)
        if key not in self.cache:
            t = torch.arange(T, device=device)
            y = torch.arange(H, device=device)
            x = torch.arange(W, device=device)
            # (t,y,x)
            grid = torch.cartesian_prod(t, y, x)        # (T*H*W, 3)
            self.cache[key] = grid
        grid = self.cache[key]                          # (T*H*W, 3)
        return grid.view(T, H * W, 3).repeat(batch_size, 1, 1).clone()


class RotaryPositionEmbedding2D_1T(nn.Module):
    """
    spatial 2D-RoPE + temporal 1D-RoPE
    tokens: (B, n_heads, T*H*W, 1024)
    positions: (B, T*H*W, 3) last dim (t,y,x)
    """
    def __init__(self, freq_spatial: float = 100.0, freq_temporal: float = 100.0):
        super().__init__()
        # self.spatial_dim = spatial_dim          # 768
        # self.temporal_dim = 1024 - spatial_dim  # 256
        self.rope_2d = RotaryPositionEmbedding2D(freq_spatial)
        self.base_freq_t = freq_temporal
        # 1D cache
        self.cache_1d: Dict[Tuple[int, torch.device, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_cos_sin_1d(self, dim: int, max_pos: int, device: torch.device, dtype: torch.dtype):
        key = (dim, max_pos, device, dtype)
        if key not in self.cache_1d:
            inv_freq = 1.0 / (self.base_freq_t ** (torch.arange(0, dim, 2, device=device).float() / dim))
            pos = torch.arange(max_pos, device=device, dtype=inv_freq.dtype)
            angle = torch.einsum("i,j->ij", pos, inv_freq)          # (max_pos, dim//2)
            angle = torch.cat([angle, angle], dim=-1)               # (max_pos, dim)
            self.cache_1d[key] = (angle.cos().to(dtype), angle.sin().to(dtype))
        return self.cache_1d[key]

    @staticmethod
    def _rotate_half(x):
        d = x.shape[-1]
        x1, x2 = x[..., :d // 2], x[..., d // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor):
        B, n_heads, N, D = tokens.shape
        assert D % 4 == 0, "Feature dimension must be divisible by 4"
        temporal_dim = D // 4
        spatial_dim = D - temporal_dim

        # 1. spatial 2D-RoPE
        spatial_tokens = tokens[..., :spatial_dim]          # (B, h, N, D // 4 * 3)
        spatial_pos = positions[..., 1:]                         # (B, N, 2)
        spatial_tokens = self.rope_2d(spatial_tokens, spatial_pos)

        # 2. temporal 1D-RoPE
        temporal_tokens = tokens[..., spatial_dim:]         # (B, h, N, D // 4)
        t_pos = positions[..., 0].long()                         # (B, N)
        max_t = int(t_pos.max()) + 1
        cos, sin = self._get_cos_sin_1d(temporal_dim, max_t, tokens.device, tokens.dtype)
        cos = F.embedding(t_pos, cos)[:, None, :, :]             # (B, 1, N, D // 4)
        sin = F.embedding(t_pos, sin)[:, None, :, :]
        temporal_tokens = (temporal_tokens * cos) + (self._rotate_half(temporal_tokens) * sin)

        # 3. concat back to D
        return torch.cat([spatial_tokens, temporal_tokens], dim=-1)
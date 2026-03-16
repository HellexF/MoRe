import os
import cv2
import numpy as np
import argparse
import torch
from searaft.config.parser import parse_args
import gc
import argparse

from searaft.core.raft import RAFT
from searaft.core.utils.flow_viz import flow_to_image
from searaft.core.utils.utils import load_ckpt
from datetime import datetime

def compute_ego_flow_batch(depths, Ks, T1s, T2s):
    N, H, W = depths.shape
    device = depths.device
    fx, fy = Ks[:, 0, 0], Ks[:, 1, 1]
    cx, cy = Ks[:, 0, 2], Ks[:, 1, 2]

    u, v = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing="xy")
    u = u.reshape(1, H, W).expand(N, -1, -1).float()
    v = v.reshape(1, H, W).expand(N, -1, -1).float()

    z = depths
    x = (u - cx[:, None, None]) * z / fx[:, None, None]
    y = (v - cy[:, None, None]) * z / fy[:, None, None]
    ones = torch.ones_like(z)
    points_cam1 = torch.stack((x, y, z, ones), dim=-1)  # (N, H, W, 4)
    points_cam1 = points_cam1.reshape(points_cam1.shape[0], -1, points_cam1.shape[-1])

    T1_inv = torch.inverse(T1s)
    points_world = torch.matmul(T1_inv, points_cam1.permute(0, 2, 1)).permute(0, 2, 1)  # (N, H, W, 4, 1)
    points_world = points_world.squeeze(-1)
    points_cam2 = torch.matmul(T2s, points_world.permute(0, 2, 1)).permute(0, 2, 1).reshape(N, H, W, 4)  # (N, H, W, 4)

    X2 = points_cam2[..., 0]
    Y2 = points_cam2[..., 1]
    Z2 = points_cam2[..., 2]
    u2 = fx[:, None, None] * X2 / Z2 + cx[:, None, None]
    v2 = fy[:, None, None] * Y2 / Z2 + cy[:, None, None]

    flow_u = u2 - u
    flow_v = v2 - v
    flow = torch.stack((flow_u, flow_v), dim=-1)
    return flow  # (N, H, W, 2)

def compute_ego_flow_batch_chunked(depths, Ks, T1s, T2s, chunk_size=1024):
    N = depths.shape[0]
    flows = []

    for i in range(0, N, chunk_size):
        d = depths[i:i+chunk_size]
        k = Ks[i:i+chunk_size]
        t1 = T1s[i:i+chunk_size]
        t2 = T2s[i:i+chunk_size]

        flow = compute_ego_flow_batch(d, k, t1, t2) 
        flows.append(flow)

    return torch.cat(flows, dim=0)


def get_mask_by_dis_threshold(seg_rgb, dis, valid, threshold):
    """
    seg_rgb: [B, H, W, 3]
    dis:     [B, H, W]
    valid:   [B, H, W]
    threshold: float
    """
    B, H, W, _ = seg_rgb.shape
    seg_flat = seg_rgb.view(B, -1, 3)       # [B, H*W, 3]
    dis_flat = dis.view(B, -1)              # [B, H*W]
    valid_flat = valid.view(B, -1)          # [B, H*W]
    
    avg_dis_dict = [{} for _ in range(B)]  # 每个 batch 一个 dict

    for b in range(B):
        mask = torch.zeros((H * W), dtype=torch.bool, device=seg_rgb.device)
        seg_b = seg_flat[b]                # [H*W, 3]
        dis_b = dis_flat[b]                # [H*W]
        valid_b = valid_flat[b]            # [H*W]

        # 找到这个 batch 的唯一颜色
        unique_colors = torch.unique(seg_b.to(torch.int32), dim=0)  # [N_color, 3]

        for color in unique_colors:
            match = (seg_b == color).all(dim=1)           # [H*W]
            matched = match & valid_b                     # [H*W]
            matched_dis = dis_b[matched]                  # [num_pixels]

            if matched_dis.numel() > 0:
                avg_dis = matched_dis.mean()
                avg_dis_dict[b][tuple(color.tolist())] = avg_dis.item()
                if avg_dis >= threshold:
                    mask[match] = True
        
        mask = mask.view(H, W).cpu().numpy()
        np.save(os.path.join(output_dir, f"motion_mask_{b:05d}.npy"), mask)
        cv2.imwrite(os.path.join(output_dir, f"motion_mask_{b:05d}.png"), mask.astype(np.uint8) * 255)
        
        del unique_colors, mask
        torch.cuda.empty_cache()

    return avg_dis_dict

def reduce_valid_stats(distances: torch.Tensor, depth_masks: torch.Tensor):
    """
    Args:
        distances: [N, H, W] float tensor
        depth_masks: [N, H, W] bool tensor

    Returns:
        count: int (有效像素总数)
        sum_: float (所有有效像素值的和)
        squared_sum: float (所有有效像素值的平方和)
    """
    masked_values = distances[depth_masks]        # [num_valid,]
    count = depth_masks.sum()
    sum_ = masked_values.sum()
    squared_sum = (masked_values ** 2).sum()
    return count, sum_, squared_sum

def run_model_in_batches(model, rgbs_tensor, rgbs_next_tensor, batch_size=16, iters=50):
    N = rgbs_tensor.shape[0]
    output_list = []

    with torch.no_grad():
        for i in range(0, N, batch_size):
            rgb_batch = rgbs_tensor[i:i + batch_size].contiguous()
            print(f"total frames: {N}, processed frames: {i}")
            rgb_next_batch = rgbs_next_tensor[i:i + batch_size].contiguous()
            out = model(rgb_batch, rgb_next_batch, iters, None, True)
            output_list.append(out['flow'][-1].cpu())
            del out
            del rgb_batch, rgb_next_batch
            torch.cuda.empty_cache()

    return torch.cat(output_list, dim=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--start', type=int, help='Which scene to process')

    # args = parser.parse_args()
    with torch.no_grad():
        dataset_dir = '/lpai/dataset/point-odyssey/0-1-0/PointOdyssey/train'
        parser.add_argument('--cfg', help='experiment configure file name', default='searaft/config/eval/kitti-L.json', type=str)
        args = parse_args(parser)

        model = RAFT(args)
        load_ckpt(model, 'searaft/ckpt/Tartan-C-T-TSKH-kitti432x960-M.pth')
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        model.eval()

        scene_list = os.listdir(dataset_dir)
        scene_list = sorted(scene_list, reverse=True)
        for scene in scene_list:
            base_path = os.path.join(dataset_dir, scene)
            if not os.path.isdir(base_path):
                continue

            output_dir = os.path.join('/lpai/volumes/base-3da-ali-sh/zhangwq/dataset', 'PointOdyssey', 'train', scene, 'masks')
            if os.path.exists(output_dir):
                continue
            print(datetime.now(), f'processing {scene}')
            anno_path = os.path.join(base_path, 'anno.npz')
            anno = np.load(anno_path)
            intrinsics, extrinsics = anno['intrinsics'], anno['extrinsics']
            step = 1

            num_frames = len(os.listdir(os.path.join(base_path, 'rgbs'))) - 1
            depths, depth_masks, rgbs, rgbs_next, masks_np = [], [], [], [], []

            for id in range(num_frames):
                depth_path = os.path.join(base_path, 'depths', f'depth_{id:05d}.png')
                rgb_path = os.path.join(base_path, 'rgbs', f'rgb_{id:05d}.jpg')
                rgb_next_path = os.path.join(base_path, 'rgbs', f'rgb_{(id + step):05d}.jpg')
                mask_path = os.path.join(base_path, 'masks', f'mask_{id:05d}.png')

                depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535. * 1000.
                mask = cv2.imread(mask_path)

                depth_mask = depth > 0.4

                depths.append(depth)
                depth_masks.append(depth_mask)
                rgbs.append(torch.from_numpy(cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)))
                rgbs_next.append(torch.from_numpy(cv2.cvtColor(cv2.imread(rgb_next_path), cv2.COLOR_BGR2RGB)))
                masks_np.append(mask)

            H, W = depths[0].shape
            depths_tensor = torch.from_numpy(np.stack(depths)).cuda()
            depth_masks_tensor = torch.from_numpy(np.stack(depth_masks)).cuda()
            rgbs_tensor = torch.stack(rgbs).permute(0, 3, 1, 2).float().cuda()
            rgbs_next_tensor = torch.stack(rgbs_next).permute(0, 3, 1, 2).float().cuda()

            # Step 1: predict flow
            output = run_model_in_batches(model, rgbs_tensor, rgbs_next_tensor, batch_size=96)
            flow_predicted = output.permute(0, 2, 3, 1).cuda()  # (N, H, W, 2)

            # Step 2: compute ego flow
            Ks = torch.from_numpy(intrinsics[:num_frames]).float().cuda()  # (N, 3, 3)
            T1s = torch.from_numpy(extrinsics[:num_frames]).float().cuda()  # (N, 4, 4)
            T2s = torch.from_numpy(extrinsics[step:num_frames + step]).float().cuda()  # (N, 4, 4)
            flow_expected = compute_ego_flow_batch_chunked(depths_tensor, Ks, T1s, T2s, 1024)

            # Step 3: fill invalid depth with predicted flow
            flow_expected[~depth_masks_tensor] = flow_predicted[~depth_masks_tensor]

            # Step 4: distance & mask
            dis = torch.norm(flow_expected - flow_predicted, dim=-1)
            masks = masks_np
            depth_masks_np = np.stack(depth_masks)

            del rgbs_tensor, rgbs_next_tensor, Ks, T1s, T2s, flow_expected, flow_predicted
            torch.cuda.empty_cache()
            count, sum_, squared_sum = reduce_valid_stats(dis, depth_masks_np)
            mean = sum_ / count
            # var = squared_sum / count - mean ** 2
            # std = torch.sqrt(var)
            threshold = mean

            os.makedirs(output_dir, exist_ok=True)

            avg_dis_dict = get_mask_by_dis_threshold(
                seg_rgb=torch.from_numpy(np.stack(masks_np)).cuda(),
                dis=dis,
                valid=depth_masks_tensor,
                threshold=mean
            )

            del depths_tensor, depth_masks_tensor, dis, output
            torch.cuda.empty_cache()
            gc.collect()

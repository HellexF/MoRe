import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import os
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
import torch
import argparse
import h5py
from PIL import Image

from searaft.config.parser import parse_args

from searaft.core.raft import RAFT
from searaft.core.utils.flow_viz import flow_to_image
from searaft.core.utils.utils import load_ckpt

def save_colored_projected_points(projected_t2, rgb, save_path="projected_t2_colored.png", stride=5):
    H, W, _ = projected_t2.shape
    u = projected_t2[..., 0]
    v = projected_t2[..., 1]

    # 下采样
    u_sampled = u[::stride, ::stride].flatten()
    v_sampled = v[::stride, ::stride].flatten()
    rgb_sampled = rgb[::stride, ::stride].reshape(-1, 3) / 255.0  # 归一化到 [0, 1]

    # 绘图
    plt.figure(figsize=(10, 8))
    plt.scatter(u_sampled, v_sampled, s=1, c=rgb_sampled)
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.title("Projected pixel positions (colored by original RGB)")
    plt.xlabel("u (x)")
    plt.ylabel("v (y)")
    plt.grid(False)
    plt.savefig(save_path, dpi=300)
    plt.close()

def create_colored_point_cloud(depth, rgb, K):
    """
    参数：
        depth: numpy.ndarray, shape=(H, W), 深度图（以米为单位）
        rgb: numpy.ndarray, shape=(H, W, 3), 对应的RGB图像（uint8）
        K: 相机内参矩阵 (3x3)，格式 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    返回：
        o3d.geometry.PointCloud 点云对象
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(u)
    pixels_homog = np.stack([u, v, ones], axis=-1)  # (H, W, 3)
    pixels_homog = pixels_homog @ np.linalg.inv(K).T  # (H, W, 3), now in camera space direction
    points_cam = pixels_homog * depth[..., np.newaxis]  # scale by depth

    # 拼接成点 (N, 3)
    xyz = points_cam.reshape(-1, 3)

    # 颜色 (N, 3)，归一化到[0, 1]
    colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0

    # 构造 Open3D 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud("colored_pointcloud.ply", pcd)
    return pcd

def compute_ego_flow(depth1, K, T1, T2):
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points_cam1 = np.stack((x, y, z, np.ones_like(z)), axis=-1).reshape(-1, 4)  # shape: (N, 4)

    points_world = (np.linalg.inv(T1) @ points_cam1.T).T  # shape: (N, 4)

    points_cam2 = (T2 @ points_world.T).T  # shape: (N, 4)
    X2, Y2, Z2 = points_cam2[:, 0], points_cam2[:, 1], points_cam2[:, 2]

    u2 = fx * X2 / Z2 + cx
    v2 = fy * Y2 / Z2 + cy

    u1 = u.flatten()
    v1 = v.flatten()
    flow_u = u2 - u1
    flow_v = v2 - v1
    flow = np.stack((flow_u, flow_v), axis=-1).reshape(H, W, 2)

    return flow

def get_mask_by_dis_threshold(seg_rgb, dis, valid, threshold):
    """
    根据距离分布筛选语义实例中距离较大的区域。

    参数：
        seg_rgb: [H, W, 3]，语义分割结果，RGB颜色表示每一类
        dis:     [H, W]，与语义分割对应的距离图（或分数图）
        valid:   [H, W]，bool 类型，表示哪些像素是有效的

    返回：
        mask: [H, W]，bool类型，表示哪些像素保留（高距离实例）
        avg_dis_dict: dict，key 为 RGB tuple，value 为该实例的平均距离
    """
    H, W, _ = seg_rgb.shape
    seg_flat = seg_rgb.reshape(-1, 3)
    dis_flat = dis.reshape(-1)
    valid_flat = valid.reshape(-1)

    # 仅在 valid 区域计算阈值
    # threshold = np.mean(valid_flat) + 2 * np.std(valid_flat)  # 可调整系数

    # outliers = dis_flat > dis_flat.mean() + 3 * dis_flat.std()

    # 遍历每个语义实例（颜色）
    unique_colors = np.unique(seg_flat, axis=0)
    mask = np.zeros((H * W,), dtype=bool)
    avg_dis_dict = {}

    for color in unique_colors:
        # if np.array_equal(color, [0, 0, 0]):
        #     continue  # 跳过背景或无效标签

        same = np.all(seg_flat == color, axis=1)
        match = same & valid_flat
        matched_dis = dis_flat[match]

        if matched_dis.size > 0:
            avg_dis = matched_dis.mean()
            avg_dis_dict[tuple(color)] = avg_dis

            if avg_dis >= threshold:
                mask[same] = True

    # mask = mask | outliers
    return mask.reshape(H, W), avg_dis_dict


def save_flow_png(flow: np.ndarray, out_path: str, max_flow: float = None):
    """
    将光流 [H, W, 2] 可视化成彩色图，并保存为 PNG。
    
    Args:
        flow: numpy.ndarray, shape (H, W, 2)，dtype float32，flow[..., 0]=u, flow[...,1]=v
        out_path: str, 要保存的 PNG 文件路径
        max_flow: float 或 None, 用于归一化速度幅值；None 表示自动按最大值缩放
    """
    # 计算幅值和角度
    u = flow[..., 0]
    v = flow[..., 1]
    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=True)

    # 创建 HSV 图像
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)  # Hue: 0–180
    hsv[..., 1] = 255  # 饱和度设为最大

    # Value 通道映射到 brightness，根据幅值归一化
    norm_mag = mag if max_flow is None else np.clip(mag / max_flow, 0, 1) * 255
    hsv[..., 2] = cv2.normalize(norm_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 转换为 BGR（OpenCV 使用），再转为 RGB 保存
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 用 Matplotlib 保存
    plt.imsave(out_path, rgb)
    print(f"✅ 光流图已保存至：{out_path}")

dataset_dir = '/lpai/dataset/point-odyssey/0-1-0/PointOdyssey/train'
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name', default='searaft/config/eval/kitti-L.json', type=str)
args = parse_args(parser)
model = RAFT(args)
load_ckpt(model, 'searaft/ckpt/Tartan-C-T-TSKH-kitti432x960-M.pth')
model = model.cuda()
model.eval()
for scene in sorted(os.listdir(dataset_dir)):
#scene = 'scene_recording_20210910_S05_S06_0_ego2'
    base_path = os.path.join('/lpai/dataset/point-odyssey/0-1-0/PointOdyssey/train', scene)
    if not os.path.isdir(base_path):
        continue
    anno_path = os.path.join(base_path, 'anno.npz')
    anno = np.load(anno_path)
    intrinsics, extrinsics = anno['intrinsics'], anno['extrinsics']
    step = 1

    distances = []
    masks = []
    depth_masks = []

    for id in range(len(os.listdir(os.path.join(base_path, 'rgbs'))) - 1):
        output_dir = os.path.join('/lpai/volumes/base-3da-ali-sh/zhangwq/dataset', 'PointOdyssey', 'train', scene, 'masks')
        os.makedirs(output_dir, exist_ok=True)
        depth_path = os.path.join(base_path, 'depths', f'depth_{id:05d}.png')
        rgb_path = os.path.join(base_path, 'rgbs', f'rgb_{id:05d}.jpg')
        rgb_next_path = os.path.join(base_path, 'rgbs', f'rgb_{(id + step):05d}.jpg')
        pose_path = os.path.join(base_path, 'flow', 'pose_left.txt')
        mask_path = os.path.join(base_path, 'masks', f'mask_{id:05d}.png')

        rgb = torch.from_numpy(cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        rgb_next = torch.from_numpy(cv2.cvtColor(cv2.imread(rgb_next_path), cv2.COLOR_BGR2RGB)).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        depth = depth.astype(np.float32) / 65535. * 1000.
        # depth = (depth.astype(np.float32) -np.min(depth)) / ((np.max(depth) - np.min(depth)) )
        # print(np.unique(depth * 255))
        # cv2.imwrite(os.path.join(output_dir, f'depth_{id:05d}.png'), (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) * 255)
        depth_mask = depth > 0.4
        # cv2.imwrite(os.path.join(output_dir, f'depth_mask_{id:05d}.png'), depth_mask.astype(np.uint8) * 255)

        # create_colored_point_cloud(depth, rgb.permute(0, 2, 3, 1).cpu().numpy(), intrinsics[id])
        mask = cv2.imread(mask_path)

        output = model(rgb, rgb_next, iters=50, test_mode=True)
        flow_predicted = output['flow'][-1].permute(0, 2, 3, 1).squeeze().detach().cpu().numpy()
        info_final = output['info'][-1]
        flow_expected = compute_ego_flow(depth, intrinsics[id], extrinsics[id], extrinsics[id + step])
        flow_expected[~depth_mask] = flow_predicted[~depth_mask]

        # # print(flow, flow_predicted)
        # save_flow_png(flow_expected, os.path.join(output_dir,f'flow_expected_{id:05d}.png'))
        # save_flow_png(flow_predicted, os.path.join(output_dir, f'flow_predicted_{id:05d}.png'))
        dis = np.linalg.norm(flow_expected - flow_predicted, axis=-1)
        distances.append(dis)
        # cv2.imwrite(os.path.join(output_dir,f"flow_diff_{id:05d}.png"), (dis - np.min(dis)) / (np.max(dis) - np.min(dis)) * 255)
        masks.append(mask)
        depth_masks.append(depth_mask)

    sum_, count = 0.0, 0.0
    squared_sum = 0.0
    for id in range(len(masks)):
        mask = depth_masks[id]
        values = distances[id][mask]
        count += mask.sum()
        sum_ += values.sum()
        squared_sum += (values ** 2).sum()

    mean = sum_ / count
    variance = (squared_sum / count) - (mean ** 2)
    std = np.sqrt(variance)
    threshold = mean

    for id in range(len(masks)):
        motion_mask, color_dict = get_mask_by_dis_threshold(masks[id], distances[id], depth_masks[id], threshold)
        np.save(os.path.join(output_dir,f"motion_mask_{id:05d}.npy"), motion_mask)
        cv2.imwrite(os.path.join(output_dir,f"motion_mask_{id:05d}.png"), motion_mask.astype(np.uint8) * 255)
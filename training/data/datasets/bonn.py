import gzip
import json
import os.path as osp
import os
import logging
from torchvision import transforms as TF
import torch
import torch.nn.functional as F
from evo.tools import file_interface

import cv2
import random
import numpy as np
import trimesh

from training.data.dataset_util import *
from training.data.base_dataset import BaseDataset

class BonnDataset(BaseDataset):
    def __init__(
        self,
        BONN_DIR: str = None,
        type_: str = "video",
    ):

        logging.info(f"BONN_DIR is {BONN_DIR}")

        self.BONN_DIR = BONN_DIR
        # self.img_size = 518
        # self.patch_size = 14
        self.training = False
        self.landscape_check = False
        self.rescale = True
        self.rescale_aug = False
        self.type = type_

        self.sequence_list = os.listdir(self.BONN_DIR)
        # self.sequence_list = ["mountain_1", "alley_1", "alley_2", "bamboo_1", "bamboo_2", "temple_2", "temple_3", "market_2", "market_5", "market_6", "cave_4", "ambush_4", "ambush_5", "ambush_6", "cave_2", "shaman_3", "sleeping_1", "sleeping_2"]
        if self.type =="mono":
            frame_list = []
            for sequence in self.sequence_list:
                for idx in range(110):
                    frame_list.append({"seq": sequence, "idx": idx})
            self.sequence_list =frame_list
        self.sequence_list_len = len(self.sequence_list)

        logging.info(f"Bonn Data size: {self.sequence_list_len}")
    
    def __len__(self):
        return len(self.sequence_list)

    def __getitem__(self, idx_N) -> dict:
        images = []
        depths = []
        motions = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        motions = []
        original_sizes = []

        if self.type == "video":
            seq_name = self.sequence_list[idx_N]
            img_path = os.path.join(self.BONN_DIR, seq_name, "rgb")
            image_list = sorted(os.listdir(img_path))
            depth_base_path = os.path.join(self.BONN_DIR, seq_name, "depth")
            depth_list = sorted(os.listdir(depth_base_path))
            traj = file_interface.read_tum_trajectory_file(os.path.join(self.BONN_DIR, seq_name, "groundtruth.txt"))
            T_wc = np.stack(traj.poses_se3)

            for frame_num in range(110):
                global_idx = 30 + frame_num
                image_path = osp.join(img_path, image_list[global_idx])
                image = read_image_cv2(image_path)
                original_size = np.array(image.shape[:2])

                depth_path = os.path.join(depth_base_path, depth_list[global_idx])
                depth_map = np.asarray(Image.open(depth_path))
                assert np.max(depth_map) > 255
                depth_map = depth_map.astype(np.float64) / 5000.0

                target_image_shape = self.get_target_shape(original_size[0] / original_size[1])

                motion = np.zeros_like(depth_map, dtype=np.int64)

                intri_opencv = np.array([[542.822841, 0.0, 315.593520],
                    [0.0, 542.576870, 237.756098],
                    [0.0, 0.0, 1.0]], dtype=np.float64)
                # extri_opencv = np.linalg.inv(T_wc[global_idx])[:3, :]
                extri_opencv = T_wc[global_idx][:3, :]

                (
                    image,
                    depth_map,
                    extri_opencv,
                    intri_opencv,
                    world_coords_points,
                    cam_coords_points,
                    point_mask,
                    *_,
                    motion,
                ) = self.process_one_image(
                    image,
                    depth_map,
                    extri_opencv,
                    intri_opencv,
                    original_size,
                    target_image_shape,
                    filepath='',
                    motion=motion,
                )

                # patch_size = 14
                # patch_scores = torch.log(1 - F.avg_pool2d(motion, kernel_size=patch_size, stride=patch_size))
                # patch_scores = patch_scores.flatten(1).cpu().numpy()

                images.append(image)
                depths.append(depth_map)
                extrinsics.append(extri_opencv)
                intrinsics.append(intri_opencv)
                cam_points.append(cam_coords_points)
                world_points.append(world_coords_points)
                point_masks.append(point_mask)
                image_paths.append(image_path)
                original_sizes.append(original_size)
            ids = range(len(os.listdir(img_path)))

        else:
            seq_name, idx = self.sequence_list[idx_N]["seq"], self.sequence_list[idx_N]["idx"]
            global_idx = 30 + idx
            image_path = osp.join(img_path, image_list[global_idx])
            image = read_image_cv2(image_path)
            original_size = np.array(image.shape[:2])

            depth_path = os.path.join(depth_base_path, depth_list[global_idx])
            depth_map = np.asarray(Image.open(filename))
            assert np.max(depth_map) > 255
            depth_map = depth_map.astype(np.float64) / 5000.0

            target_image_shape = self.get_target_shape(original_size[0] / original_size[1])

            motion = np.zeros_like(depth_map, dtype=np.int64)

            intri_opencv = np.array([[542.822841, 0.0, 315.593520],
                [0.0, 542.576870, 237.756098],
                [0.0, 0.0, 1.0]], dtype=np.float64)
            extri_opencv = np.linalg.inv(T_wc[global_idx])[:3, :]

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                *_,
                motion,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath='',
                motion=motion,
            )

            images.append(image)
            depths.append(depth_map)
            motions.append(motion)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            image_paths.append(image_path)
            original_sizes.append(original_size)
            ids = [idx]

        set_name = "bonn"
        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "motions": motions,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }

        # colored_points = []
        # colored_colors = []

        # for pts, img, mask in zip(world_points, images, point_masks):
        #     pts = np.asarray(pts)
        #     img = np.asarray(img)

        #     # 如果图像值是 0-255，需要归一化到 0-1
        #     if img.max() > 1.0:
        #         img = img / 255.0

        #     # 如果 pts 是 [H, W, 3]，展开成 [N, 3]
        #     if len(pts.shape) == 3:
        #         pts = pts.reshape(-1, 3)
        #     if len(img.shape) == 3:
        #         colors = img.reshape(-1, 3)

        #     # 只保留有效点（非 NaN 或非 inf）
        #     valid_mask = mask.reshape(-1, )
        #     pts = pts[valid_mask]
        #     colors = colors[valid_mask]

        #     # 收集到大数组
        #     colored_points.append(pts)
        #     colored_colors.append(colors)

        # # 合并多帧点云
        # all_points = np.vstack(colored_points)
        # all_colors = np.vstack(colored_colors)

        # # trimesh 需要颜色是 uint8 [0-255]
        # all_colors = (all_colors * 255).astype(np.uint8)

        # # 创建点云并保存
        # pcd = trimesh.points.PointCloud(all_points, colors=all_colors)
        # if not os.path.exists(f"colored_pointcloud_{seq_name}.ply"):
        #     pcd.export(f"colored_pointcloud_{seq_name}.ply")

        #     print(f"PLY 文件已保存为 colored_pointcloud_{seq_name}.ply")
        return batch

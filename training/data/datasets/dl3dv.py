import gzip
import json
import os.path as osp
import os
import glob
import logging

import cv2
import random
import numpy as np
import torch
import re

from data.dataset_util import *
from data.base_dataset import BaseDataset
import trimesh

class DL3DVDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        DL3DV_DIR: str = None,
        min_num_images: int = 24    ,
        len_train: int = 2000,
        len_test: int = 1000,
    ):
        """
        Initialize the HOI4DDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            DL3DV_DIR (str): Directory path to Dynamic Replica data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If DL3DV_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if DL3DV_DIR is None:
            raise ValueError("DL3DV_DIR must be specified.")

        if split == "train":
            split_name_list = ["train"]
            self.len_train = len_train
        elif split == "test":
            split_name_list = ["test"]
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}") 

        self.sequence_list = [seq for seq in os.listdir(DL3DV_DIR) if os.path.isdir(os.path.join(DL3DV_DIR, seq, seq)) and (seq != "d039dadf4a29e0cb81ace31443d69c85d75d686c87605359f101ec3684c5bcd2" and seq != "dbe5cf545498826740f028ec0f05fd3cf844c38a383c60d1e9126955245123f3" and seq != "b2ddf3abc829eb768845d960e00b69b9bca1bc216fc1402590f969a231460e92")]
        if split == "test":
            self.sequnce_list = self.sequence_list[-10:]
        if self.debug:
            self.sequence_list = self.sequence_list[:3]
        self.split = split   
        
        self.seqlen = None
        self.min_num_images = min_num_images
        self.seqlen = None

        logging.info(f"DL3DV_DIR is {DL3DV_DIR}")
        self.DL3DV_DIR = DL3DV_DIR

        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = 0
        # for seq_name in self.sequence_list:
        #     self.total_frame_num += len(os.listdir(osp.join(self.DL3DV_DIR, seq_name, seq_name, "images_4")))

        status = "Training" if self.training else "Test"
        logging.info(f"{status}: DL3DV Data size: {self.sequence_list_len}")
        logging.info(f"{status}: DL3DV Data dataset length: {len(self)}")

    def get_data(
            self,
            seq_index: int = None,
            img_per_seq: int = None,
            seq_name: str = None,
            ids: list = None,
            aspect_ratio: float = 1.0,
        ) -> dict:
            """
            Retrieve data for a specific sequence.

            Args:
                seq_index (int): Index of the sequence to retrieve.
                img_per_seq (int): Number of images per sequence.
                seq_name (str): Name of the sequence.
                ids (list): Specific IDs to retrieve.
                aspect_ratio (float): Aspect ratio for image processing.

            Returns:
                dict: A batch of data including images, depths, and other metadata.
            """
            
            seq_index = seq_index % len(self.sequence_list)
            if seq_name is None:
                seq_name = self.sequence_list[seq_index]

            target_image_shape = self.get_target_shape(aspect_ratio)

            images = []
            depths = []
            cam_points = []
            world_points = []
            point_masks = []
            extrinsics = []
            intrinsics = []
            image_paths = []
            original_sizes = []
            motions = []
            
            base_frame_path = os.path.join(self.DL3DV_DIR, seq_name, seq_name, "images_4")

            with open(osp.join(self.DL3DV_DIR, seq_name, seq_name, "transforms.json"), "r") as f:
                data = json.load(f)

            fl_x, fl_y = data["fl_x"], data["fl_y"]
            cx, cy = data["cx"], data["cy"]
            intri_opencv = np.array([
                [fl_x * 0.25, 0,    cx * 0.25],
                [0,    fl_y * 0.25, cy * 0.25],
                [0,    0,    1]
            ], dtype=np.float64)
            num_frames = len(data["frames"])

            if ids is not None:
                start_idx = ids[0]
            elif img_per_seq is not None:
                if self.debug:
                    ids = np.arange(24)
                else:
                    ids = self.sample_fixed_interval_ids(num_frames, img_per_seq)
                start_idx = ids[0]
            else:
                start_idx = 0
                img_per_seq = num_frames
                ids = np.arange(num_frames)
                 

            for global_idx in ids:
                image_path = osp.join(base_frame_path, data["frames"][global_idx]["file_path"][7:])
                image = read_image_cv2(image_path)

                if image is None:
                    image_path = osp.join(base_frame_path, data["frames"][global_idx + 1]["file_path"][7:])
                    image = read_image_cv2(image_path)
                depth_map = np.full(image.shape[:2], -1.0, dtype=float)

                motion = np.zeros_like(depth_map)

                original_size = np.array(image.shape[:2])

                transform = np.array(data["frames"][global_idx]["transform_matrix"], dtype=np.float64)
                extri_opencv = transform[:3, :]
                t = extri_opencv[:, 3]
                t_norm = np.linalg.norm(t)
                extri_opencv[:, 3] = t / t_norm

                (
                    image,
                    depth_map,
                    extri_opencv,
                    intri_opencv,
                    world_coords_points,
                    cam_coords_points,
                    point_mask,
                    _,
                    motion
                ) = self.process_one_image(
                    image,
                    depth_map,
                    extri_opencv,
                    intri_opencv,
                    original_size,
                    target_image_shape,
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

            set_name = "dl3dv"

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

            # extrinsics, cam_points, world_points, depth_gt = \
            # normalize_camera_extrinsics_and_points_batch(
            #     extrinsics=torch.from_numpy(np.stack(extrinsics, axis=0)).unsqueeze(0),
            #     cam_points=torch.from_numpy(np.stack(cam_points, axis=0)).unsqueeze(0),
            #     world_points=torch.from_numpy(np.stack(world_points, axis=0)).unsqueeze(0),
            #     depths=torch.from_numpy(np.stack(depths, axis=0)).unsqueeze(0),
            #     point_masks=torch.from_numpy(np.stack(point_masks, axis=0)).unsqueeze(0),
            # )
            
            # world_points = world_points.squeeze(0)
            # colored_points = []
            # colored_colors = []

            # for pts, img in zip(world_points, images):
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
            #     valid_mask = np.isfinite(pts).all(axis=1)
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
            # seq_name = seq_name.replace("/", "-")
            # if not os.path.exists(f"colored_pointcloud_{seq_name}.ply"):
            #     pcd.export(f"colored_pointcloud_{seq_name}.ply")

            #     print(f"PLY 文件已保存为 colored_pointcloud_{seq_name}.ply")

            return batch
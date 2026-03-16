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
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
import trimesh

class UnrealStereo4kDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        UNREALSTEREO4K_DIR: str = None,
        min_num_images: int = 24    ,
        len_train: int = 1000,
        len_test: int = 1000,
    ):
        """
        Initialize the Unreal4kStereo.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            UNREALSTEREO4K_DIR (str): Directory path to Unreal4kStereo data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If UNREALSTEREO4K_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if UNREALSTEREO4K_DIR is None:
            raise ValueError("UNREALSTEREO4K_DIR must be specified.")

        if split == "train":
            split_name_list = ["train"]
            self.len_train = len_train
        elif split == "test":
            split_name_list = ["test"]
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}") 

        self.sequence_list = [
            os.path.relpath(p, UNREALSTEREO4K_DIR)
            for p in glob.glob(os.path.join(UNREALSTEREO4K_DIR, "*", "*"))
            if os.path.isdir(p)
        ]
        if split == "test":
            self.sequnce_list = self.sequence_list[-10:]
        if self.debug:
            self.sequence_list = self.sequence_list[:3]
        self.split = split   
        
        self.seqlen = None
        self.min_num_images = min_num_images

        logging.info(f"UNREALSTEREO4K_DIR is {UNREALSTEREO4K_DIR}")
        self.UNREALSTEREO4K_DIR = UNREALSTEREO4K_DIR

        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = 0
        for seq_name in self.sequence_list:
            self.total_frame_num += len(os.listdir(os.path.join(self.UNREALSTEREO4K_DIR, seq_name))) // 3

        # if self.eval:
        #     if self.training:
        #         self.len_train = self.sequence_list_len
        #     else:
        #         self.len_test = self.sequence_list_len
        status = "Training" if self.training else "Test"
        logging.info(f"{status}: UNREALSTEREO4K Data size: {self.sequence_list_len}")
        logging.info(f"{status}: UNREALSTEREO4K Data dataset length: {len(self)}")

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
            
            base_frame_path = os.path.join(self.UNREALSTEREO4K_DIR, seq_name)
            frame_list = os.listdir(base_frame_path)
            num_frames = len(frame_list) // 3
            if ids is not None:
                start_idx = ids[0]
            elif img_per_seq is not None:
                if self.debug:
                    ids = np.arange(120)
                else:
                    ids = self.sample_fixed_interval_ids(num_frames, img_per_seq)
                start_idx = ids[0]
            else:
                start_idx = 0
                img_per_seq = num_frames
                ids = np.arange(num_frames)
                
            for global_idx in ids:
                image_path = osp.join(base_frame_path, f"{global_idx:05d}_rgb.png")
                image = read_image_cv2(image_path)

                depth_path = os.path.join(base_frame_path, f"{global_idx:05d}_depth.npy")
                depth_map = np.load(depth_path)
                depth_map[depth_map > 40] = 0.0

                motion = np.zeros_like(depth_map)

                original_size = np.array(image.shape[:2])
                cam_path = osp.join(base_frame_path, f"{global_idx:05d}.npz")
                cam_file = np.load(cam_path)
                intri_opencv = cam_file["intrinsics"]
                extri_opencv = cam_file["cam2world"][:3, :]

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

            set_name = "unreal4dstereo"

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

            extrinsics, cam_points, world_points, depth_gt = \
            normalize_camera_extrinsics_and_points_batch(
                extrinsics=torch.from_numpy(np.stack(extrinsics, axis=0)).unsqueeze(0),
                cam_points=torch.from_numpy(np.stack(cam_points, axis=0)).unsqueeze(0),
                world_points=torch.from_numpy(np.stack(world_points, axis=0)).unsqueeze(0),
                depths=torch.from_numpy(np.stack(depths, axis=0)).unsqueeze(0),
                point_masks=torch.from_numpy(np.stack(point_masks, axis=0)).unsqueeze(0),
            )
            
            world_points = world_points.squeeze(0)

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
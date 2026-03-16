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

class MatrixCityDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        MATRIXCITY_DIR: str = None,
        min_num_images: int = 24    ,
        len_train: int = 5000,
        len_test: int = 1000,
    ):
        """
        Initialize the MatrixCityDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            MATRIX_CITY_DIR (str): Directory path to MatrixCity data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If MATRIXCITY_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if MATRIXCITY_DIR is None:
            raise ValueError("MATRIXCITY_DIR must be specified.")

        if split == "train":
            split_name_list = ["train"]
            self.len_train = len_train
        elif split == "test":
            split_name_list = ["test"]
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}") 

        self.sequence_list = ["small_city_road_down", "small_city_road_horizon", "small_city_road_outside", "small_city_road_vertical"]
        if self.debug:
            self.sequence_list = self.sequence_list[:3]
        self.split = split   
        
        self.seqlen = None
        self.min_num_images = min_num_images
        self.seqlen = None

        logging.info(f"MATRIXCITY_DIR is {MATRIXCITY_DIR}")
        self.MATRIXCITY_DIR = MATRIXCITY_DIR

        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = 0

        # if self.eval:
        #     if self.training:
        #         self.len_train = self.sequence_list_len
        #     else:
        #         self.len_test = self.sequence_list_len
        status = "Training" if self.training else "Test"
        logging.info(f"{status}: MatrixCity Data size: {self.sequence_list_len}")
        logging.info(f"{status}: MatrixCity Data dataset length: {len(self)}")
        logging.info(f"{status}: MatrixCity Data frame nums: {self.total_frame_num}")

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
            
            base_frame_path = os.path.join(self.MATRIXCITY_DIR, "small_city", "street", self.split, seq_name)
            base_depth_path = os.path.join(self.MATRIXCITY_DIR, "small_city_depth", "street", self.split, f"{seq_name}_depth")
            cam_path = os.path.join(base_frame_path, "transforms.json")
            with open(cam_path, "r") as f:
                tj = json.load(f)
            frames = tj['frames']
            num_frames = len(frames)
            angle_x = tj['camera_angle_x']
            w = float(1000)
            h = float(1000)
            fl_x = float(.5 * w / np.tan(.5 * angle_x))
            fl_y = fl_x
            cx = w / 2
            cy = h / 2
            intri = np.array([[fl_x,0,cx],[0,fl_y,cy],[0,0,1]])
            R_b2cv = np.array([
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0]
            ])
            if ids is not None:
                start_idx = ids[0]
            elif img_per_seq is not None:
                if self.debug:
                    ids = np.arange(4)
                else:
                    ids = self.sample_fixed_interval_ids(num_frames, img_per_seq, max_gap=1)
                start_idx = ids[0]
            else:
                start_idx = 0
                img_per_seq = num_frames
                ids = np.arange(num_frames)
                 
            for global_idx in ids:
                frame = frames[global_idx]
                idx = frame["frame_index"]
                image_path = osp.join(base_frame_path, f"{idx:04d}.png")
                image = read_image_cv2(image_path)
                
                depth_path = osp.join(base_depth_path, f"{idx:04d}.exr")
                depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[...,0] #(H, W)
                invalid_mask=(depth_map==65504)
                depth_map[invalid_mask] = 0.0
                depth_map = depth_map / 100  # cm -> m

                motion = np.zeros_like(depth_map)

                original_size = np.array(image.shape[:2])
                c2w = np.array(frame['rot_mat'])
                c2w[:3,:3] *= 100
                c2w[:3,3] /= 100
                # R_b = c2w[:3, :3]
                # t_b = c2w[:3, 3]

                # R_cv = R_b2cv @ R_b @ R_b2cv.T
                # t_cv = R_b2cv @ t_b

                # extri_opencv = np.eye(4)
                # extri_opencv[:3, :3] = R_cv
                # extri_opencv[:3, 3] = t_cv
                # extri_opencv = c2w[:3, :]
                # extri_opencv = np.linalg.inv(extri_opencv)[:3, :]
                extri_opencv = np.linalg.inv(c2w)[:3, :]
                intri_opencv = intri

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

            set_name = "MatrixCity"

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
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

class ARKitScenesDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        ARKITSCENES_DIR: str = None,
        min_num_images: int = 24    ,
        len_train: int = 5000,
        len_test: int = 1000,
    ):
        """
        Initialize the ARKitScenesDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            ARKITSCENES_DIR (str): Directory path to Dynamic Replica data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If ARKITSCENES_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if ARKITSCENES_DIR is None:
            raise ValueError("ARKITSCENES_DIR must be specified.")

        if split == "train":
            split_name_list = ["train"]
            self.split = "Training"
            self.len_train = len_train
        elif split == "test":
            split_name_list = ["test"]
            self.split = "Validation"
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}") 

        self.cam_info = {}
        self.sequence_list = []
        for seq in os.listdir(os.path.join(ARKITSCENES_DIR, self.split)):
            if osp.isdir(osp.join(ARKITSCENES_DIR, self.split, seq)):
                self.sequence_list.append(seq)
                with np.load(osp.join(ARKITSCENES_DIR, self.split, seq, "scene_metadata.npz")) as data:
                    imgs_with_indices = sorted(
                        enumerate(data["images"]), key=lambda x: x[1]
                    )
                    indices = [x[0] for x in imgs_with_indices]
                    num_imgs = len(os.listdir(osp.join(ARKITSCENES_DIR, self.split, seq, "vga_wide",)))
                    K = np.expand_dims(np.eye(3), 0).repeat(num_imgs, 0)
                    intrins = data["intrinsics"][indices]
                    K[:, 0, 0] = [fx for _, _, fx, _, _, _ in intrins]
                    K[:, 1, 1] = [fy for _, _, _, fy, _, _ in intrins]
                    K[:, 0, 2] = [cx for _, _, _, _, cx, _ in intrins]
                    K[:, 1, 2] = [cy for _, _, _, _, _, cy in intrins]
                    self.cam_info[seq] = {"intri": list(K), "extri": list(data["trajectories"][indices])}

        self.seqlen = None
        self.cameras_map = [[] for _ in self.sequence_list]
        self.min_num_images = min_num_images
        self.seqlen = None

        logging.info(f"ARKITSCENES_DIR is {ARKITSCENES_DIR}")
        self.ARKITSCENES_DIR = ARKITSCENES_DIR

        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = 0
        for seq_name in self.sequence_list:
            self.total_frame_num += len(os.listdir(osp.join(self.ARKITSCENES_DIR, self.split, seq)))

        # if self.eval:
        #     if self.training:
        #         self.len_train = self.sequence_list_len
        #     else:
        #         self.len_test = self.sequence_list_len
        status = "Training" if self.training else "Test"
        logging.info(f"{status}: ARKitScenes Data size: {self.sequence_list_len}")
        logging.info(f"{status}: ARKitScenes Data dataset length: {len(self)}")
        logging.info(f"{status}: ARKitScenes Data frame nums: {self.total_frame_num}")

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
            
            base_frame_path = os.path.join(self.ARKITSCENES_DIR, self.split, seq_name, "vga_wide")
            base_depth_path = os.path.join(self.ARKITSCENES_DIR, self.split, seq_name, "highres_depth")

            frame_list = sorted(os.listdir(base_frame_path))
            num_frames = len(frame_list)
            base_num_frames = num_frames
            if ids is not None:
                start_idx = ids[0]
            elif img_per_seq is not None:
                if self.debug:
                    ids = np.arange(24)
                else:
                    while(num_frames < img_per_seq):
                        num_frames *= 2
                    ids = self.sample_fixed_interval_ids(num_frames, img_per_seq, max_gap=1)
                start_idx = ids[0]
            else:
                start_idx = 0
                img_per_seq = num_frames
                ids = np.arange(num_frames)
                 

            for global_idx in ids:
                idx = global_idx % base_num_frames
                image_path = osp.join(base_frame_path, frame_list[idx])
                image = read_image_cv2(image_path)

                depth_path = os.path.join(base_depth_path, frame_list[idx].replace("jpg", "png"))
                depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                depth_map = depth_map.astype(np.float32) / 1000.0
                depth_map[~np.isfinite(depth_map)] = 0
                depth_map[depth_map > 80] = 0

                motion = np.zeros_like(depth_map)

                original_size = np.array(image.shape[:2])

                intri_opencv = self.cam_info[seq_name]["intri"][idx]
                extri_opencv = np.linalg.inv(self.cam_info[seq_name]["extri"][idx])[:3, :]

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

            set_name = "arkitscenes"

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
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import logging
import random
import glob
import json
import imageio
from scipy.spatial.transform import Rotation as R
import trimesh

import cv2
import numpy as np

import sys
sys.path.append('/mnt/volumes/base-3da-ali-sh-mix/zhangwq/vggt/training')
from data.dataset_util import *
from data.base_dataset import BaseDataset

class OmniWorldDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        OMNIWORLD_DIR: str = "/lpai/volumes/base-3da-ali-sh-mix/zhangwq/dataset/vggt/OmniWorld",
        min_num_images: int = 24,
        len_train: int = 5000,
        len_test: int = 1000,
        expand_ratio: int = 8,
    ):
        """
        Initialize the VKittiDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            VKitti_DIR (str): Directory path to VKitti data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
            expand_range (int): Range for expanding nearby image selection.
            get_nearby_thres (int): Threshold for nearby image selection.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        
        self.expand_ratio = expand_ratio
        self.OMNIWORLD_DIR = OMNIWORLD_DIR
        self.min_num_images = min_num_images

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        
        logging.info(f"OMNIWORLD_DIR is {self.OMNIWORLD_DIR}")

        # sequence_list = []
        # self.seq_info = {}
        # self.total_frame_num = 0
        # for scene in os.listdir(self.OMNIWORLD_DIR):
        #     if not osp.exists(osp.join(self.OMNIWORLD_DIR, scene, "camera")) or not osp.exists(osp.join(self.OMNIWORLD_DIR, scene, "0000")):
        #         continue
        #     scene_path = os.path.join(self.OMNIWORLD_DIR, scene)
        #     count = 0
        #     split_info_path = os.path.join(scene_path, "split_info.json")
        #     with open(split_info_path, "r", encoding="utf-8") as f:
        #         split_info = json.load(f)
        #     split_num = split_info["split_num"]
        #     num_per_seq = len(os.listdir(osp.join(self.OMNIWORLD_DIR, scene, f"0000", "color")))
        #     for idx in range(split_num):
        #         seq_name = f"{scene}_{idx:04d}"
        #         end_idx = split_info["split"][idx][-1]
        #         set_num = end_idx // num_per_seq
        #         if not osp.exists(osp.join(self.OMNIWORLD_DIR, scene, f"{set_num:04d}", "color", f"{end_idx:06d}.png")):
        #             break
        #         info = split_info["split"][idx]
        #         len_info = len(info)
        #         if (len_info < 36):
        #             continue
        #         self.total_frame_num += len_info
        #         sequence_list.append(seq_name)
        #         self.seq_info[seq_name] = info
        # self.sequence_list = sequence_list
        with open("/lpai/volumes/base-3da-ali-sh-mix/zhangwq/dataset/vggt/omniworld_sequences.json", "r", encoding="utf-8") as f:
                data = json.load(f)
        self.sequence_list = data["sequence_list"]
        self.seq_info = data["seq_info"]

        # sequence_list_path = osp.join(self.OMNIWORLD_DIR, "sequence.json")
        # if os.path.exists(sequence_list_path):
        #     with open(path, "r", encoding="utf-8") as f:
        #         sequence_list = json.load(f)
        # else:
        #     parent = os.path.dirname(sequence_list_path)
        #     if parent and not os.path.exists(parent):
        #         os.makedirs(parent, exist_ok=True)
        #         with open(path, "w", encoding="utf-8") as f:
        #             json.dump(sequence_list, f, ensure_ascii=False, indent=2)

        self.sequence_list_len = len(self.sequence_list)

        if self.debug:
            # self.sequence_list = self.sequence_list[:3]
            self.sequence_list = ['3aa5666fe91c_0000']

        self.depth_max = 80

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: OmniWorld Real Data size: {self.sequence_list_len}")
        logging.info(f"{status}: OmniWorld Data dataset length: {len(self)}")
        # logging.info(f"{status}: OmniWorld Data frame nums: {self.total_frame_num}")

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

        info = self.seq_info[seq_name]
        num_images = len(info)
        base_index = info[0]
        seq, idx = seq_name.split("_", 1)

        if ids is not None:
            start_idx = ids[0]
        elif img_per_seq is not None:
            if self.debug:
                ids = np.arange(24)
            else:
                ids = self.sample_fixed_interval_ids(num_images, img_per_seq)
            start_idx = ids[0]
        else:
            if self.debug:
                ids = np.arange(24)
            else:
                start_idx = 0
                img_per_seq = num_images
                ids = np.arange(num_images)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []
        motions = []

        def load_depth(depthpath):
            """
            Returns
            -------
            depthmap : (H, W) float32
            valid   : (H, W) bool      True for reliable pixels
            """

            depthmap = imageio.v2.imread(depthpath).astype(np.float32) / 65535.0
            # breakpoint()
            near_mask = depthmap < (150 / 65535.0)   # 1. too close
            far_mask = depthmap >= (65500.0 / 65535.0) # 2. filter sky
            far_mask = depthmap > np.percentile(depthmap[~far_mask], 80) # 3. filter far area (optional)
            near, far = 1., 1000.
            depthmap = depthmap / (far - depthmap * (far - near)) / 0.004

            valid = ~(near_mask | far_mask)
            depthmap[~valid] = 0.

            return depthmap, valid

        num_per_seq = len(os.listdir(osp.join(self.OMNIWORLD_DIR, seq, f"0000", "color")))
        for image_idx in ids:
            orig_idx = image_idx
            image_idx += base_index
            index = image_idx // num_per_seq
            image_filepath = osp.join(self.OMNIWORLD_DIR, seq, f"{index:04d}", "color", f"{image_idx:06d}.png")
            depth_filepath = osp.join(self.OMNIWORLD_DIR, seq, f"{index:04d}", "depth", f"{image_idx:06d}.png")

            image = read_image_cv2(image_filepath)
            if not os.path.exists(depth_filepath):
                raise FileNotFoundError(f"Depth file not found: {depth_filepath}")
            depth_map, _ = load_depth(depth_filepath)
            depth_map = threshold_depth_map(depth_map, max_percentile=-1, min_percentile=-1, max_depth=self.depth_max)

            assert image.shape[:2] == depth_map.shape, f"Image and depth shape mismatch: {image.shape[:2]} vs {depth_map.shape}"

            motion_path = osp.join(self.OMNIWORLD_DIR, seq, "gdino_mask", f"{image_idx:06d}.png")
            if not osp.exists(motion_path):
                logging.warning(f"Motion mask file not found: {motion_path}. Using empty mask.")
                motion = np.zeros_like(depth_map, dtype=np.int64)
            else:
                motion = Image.open(motion_path).convert('L')
                motion = np.array(motion )  # 0..255
                motion = (motion  >= 128).astype(np.int64) 

            original_size = np.array(image.shape[:2])

            # Process camera matrices
            cam_file = os.path.join(self.OMNIWORLD_DIR, seq, "camera", f"split_{int(idx)}.json")
            with open(cam_file, "r", encoding="utf-8") as f:
                cam = json.load(f)

            extri_opencv = np.eye(4)

            quat_wxyz = np.array(cam["quats"][orig_idx])           # (S, 4)  (w,x,y,z)
            quat_xyzw = np.roll(quat_wxyz, shift=-1, axis=-1)

            rotation = R.from_quat(quat_xyzw).as_matrix()
            translation = np.array(cam["trans"][orig_idx])

            extri_opencv[:3, :3] = rotation
            extri_opencv[:3, 3] = translation
            extri_opencv = extri_opencv[:3, :]
            # extri_opencv = np.linalg.inv(extri_opencv)[:3, :]

            intri_opencv = np.eye(3)
            intri_opencv[0, 0] = cam["focals"][orig_idx]
            intri_opencv[1, 1] = cam["focals"][orig_idx]
            intri_opencv[0, 2] = cam["cx"]
            intri_opencv[1, 2] = cam["cy"]

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
                filepath=image_filepath,
                motion=motion
            )

            if (image.shape[:2] != target_image_shape).any():
                logging.error(f"Wrong shape for {seq_name}: expected {target_image_shape}, got {image.shape[:2]}")
                continue

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)
            motions.append(motion)

        set_name = "omniworld"
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
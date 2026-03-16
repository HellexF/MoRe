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
import torch
import trimesh

import cv2
import numpy as np

from data.dataset_util import *
from data.base_dataset import BaseDataset
from scipy.spatial.transform import Rotation as R
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch


class TartanDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        TARTAN_DIR: str = "/lpai/dataset/action-model-group-data-weight/0-3-0/TartanAir/dataset",
        TARTAN_LIST_DIR: str = "/lpai/volumes/base-3da-ali-sh-mix/zhangwq/dataset/vggt/TartanAir",
        TARTAN_MOTION_DIR: str ="",
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
            Tartan_DIR (str): Directory path to VKitti data.
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
        self.TARTAN_DIR = TARTAN_DIR
        self.TARTAN_LIST_DIR = TARTAN_LIST_DIR
        self.TARTAN_MOTION_DIR = TARTAN_MOTION_DIR
        self.min_num_images = min_num_images

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            split = 'val'
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        
        logging.info(f"TARTAN_DIR is {self.TARTAN_DIR}")

        self.cam_info = {}
        sequence_list = []
        self.total_frame_num = 0
        # Load or generate sequence list
        txt_path = osp.join(self.TARTAN_LIST_DIR, f"{split}_sequence_list.txt")
        if osp.exists(txt_path):
            with open(txt_path, 'r') as f:
                sequence_list = [line.strip() for line in f.readlines()]
        else:
            # Generate sequence list and save to txt            
            for scene in os.listdir(os.path.join(self.TARTAN_DIR, 'rgb')):
                for dif in os.listdir(os.path.join(self.TARTAN_DIR, 'rgb', scene)):
                    for seq in os.listdir(os.path.join(self.TARTAN_DIR, 'rgb', scene, dif)):
                        for cam in os.listdir(os.path.join(self.TARTAN_DIR, 'rgb', scene, dif, seq)):
                            if not 'left' in cam:
                                continue
                            seq_name = os.path.join(scene, dif, seq, cam)
                            if os.path.isdir(os.path.join(self.TARTAN_DIR, 'rgb', seq_name)):
                                sequence_list.append(seq_name)
                
            # Save to txt file
            with open(txt_path, 'w') as f:
                f.write('\n'.join(sequence_list))

        for seq_name in sequence_list:
            extrinsics = []
            with open(os.path.join(self.TARTAN_DIR, 'rgb',seq_name.replace('image', 'pose') + '.txt'), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 7:
                        numbers = list(map(float, parts))
                        extrinsics.append(numbers)
            self.cam_info[seq_name] = {}
            self.cam_info[seq_name]['intrinsic'] = np.array([
                [320.0, 0., 320.0],
                [0., 320.0, 240.0],
                [0., 0., 1.]
            ])
            self.cam_info[seq_name]['extrinsic'] = []
            self.total_frame_num += len(extrinsics)
            for idx in range(len(extrinsics)):
                t, q = extrinsics[idx][:3], extrinsics[idx][3:]
                r = R.from_quat(q)
                R_mat = r.as_matrix()
                # r = np.zeros((3, 3), dtype = np.float64)
                # r[0, 0] =  1.0
                # r[1, 2] =  1.0
                # r[2, 1] = -1.0

                # R_mat = np.matmul(np.matmul(r, R_mat), r.transpose())
                # t = r.dot(t) 

                # T = np.eye(4)
                # T[:3, :3] = R_mat
                # T[:3, 3] = t
                c2w = np.eye(4)
                c2w[:3, :3] = R_mat
                c2w[:3, 3] = t
                w2c = np.linalg.inv(c2w)
                w2c = w2c[[1, 2, 0, 3]]
                T = w2c
                # T = np.linalg.inv(T)
                self.cam_info[seq_name]['extrinsic'].append(T)

        self.sequence_list = sequence_list
        self.sequence_list_len = len(self.sequence_list)

        self.depth_max = 80

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: Tartan Real Data size: {self.sequence_list_len}")
        logging.info(f"{status}: Tartan Data dataset length: {len(self)}")
        logging.info(f"{status}: Tartan Data frame nums: {self.total_frame_num}")

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

        image_dir = os.path.join(self.TARTAN_DIR, 'rgb', seq_name)
        num_frames = len(os.listdir(image_dir))
        if ids is not None:
            start_idx = ids[0]
        elif img_per_seq is not None:
            ids = self.sample_fixed_interval_ids(num_frames, img_per_seq)
            start_idx = ids[0]
        else:
            start_idx = 0
            img_per_seq = num_frames
            ids = np.arange(num_frames)

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

        for image_idx in ids:
            image_filepath = osp.join(self.TARTAN_DIR, 'rgb', seq_name, f"{image_idx:06d}_{seq_name.rpartition('_')[2]}.png")
            depth_filepath = osp.join(self.TARTAN_DIR, 'depth', seq_name.replace('image', 'depth'), f"{image_idx:06d}_{seq_name.rpartition('_')[2]}_depth.npy")

            image = read_image_cv2(image_filepath)
            depth_map = np.load(depth_filepath)
            depth_map = threshold_depth_map(depth_map, max_percentile=-1, min_percentile=-1, max_depth=self.depth_max)

            assert image.shape[:2] == depth_map.shape, f"Image and depth shape mismatch: {image.shape[:2]} vs {depth_map.shape}"

            motion_path = osp.join(self.TARTAN_MOTION_DIR, seq_name.replace('image', 'mask'), f"{image_idx:06d}_mask.npy")
            if not osp.exists(motion_path):
                logging.warning(f"Motion mask file not found: {motion_path}. Using empty mask.")
                motion = np.zeros_like(depth_map, dtype=np.int64)
            else:
                try:
                    motion = np.load(motion_path).astype(np.int64)
                except EOFError:
                    motion = np.zeros_like(depth_map, dtype=np.int64)

            original_size = np.array(image.shape[:2])

            # Process camera matrices
            extri_opencv = self.cam_info[seq_name]['extrinsic'][image_idx][:3, :]

            intri_opencv = self.cam_info[seq_name]['intrinsic']

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

        set_name = "tartan air"
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
        # seq_name = seq_name.replace('/', '-')
        # if not os.path.exists(f"colored_pointcloud_{seq_name}.ply"):
        #     pcd.export(f"colored_pointcloud_{seq_name}.ply")

        #     print(f"PLY 文件已保存为 colored_pointcloud_{seq_name}.ply")
        return batch
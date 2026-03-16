import gzip
import json
import os.path as osp
import os
import logging
import torch

import cv2
import random
import numpy as np

import sys
sys.path.append('/mnt/volumes/base-3da-ali-sh-mix/zhangwq/vggt/training')
from data.dataset_util import *
from data.base_dataset import BaseDataset
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
import trimesh

class SpringDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        SPRING_DIR: str = None,
        SPRING_ANNO_DIR: str = None,
        min_num_images: int = 13,
        len_train: int = 5000,
        len_test: int = 500,
    ):
        """
        Initialize the SpringDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            SPRING_DIR (str): Directory path to Spring data.
            SPRING_ANNO_DIR (str): Directory path to processed Spring data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If SPRING_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if SPRING_DIR is None or SPRING_ANNO_DIR is None:
            raise ValueError("Both SPRING_DIR and SPRING_ANNO_DIR must be specified.")

        if self.debug:
            self.sequence_list = ['0001', '0002', '0008']
        else:
            self.sequence_list = sorted(os.listdir(os.path.join(SPRING_DIR, f'{split}_frame_left', 'spring', split)))
            self.sequence_list = self.sequence_list
        self.split = split

        if split == "train":
            split_name_list = ["train"]
            self.len_train = len_train
        elif split == "test":
            split_name_list = ["test"]
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")    
        
        self.seqlen = None
        self.cameras_map = [[] for _ in self.sequence_list]
        self.min_num_images = min_num_images
        self.seqlen = None

        logging.info(f"SPRING_DIR is is {SPRING_DIR}")
        self.SPRING_DIR = SPRING_DIR
        self.SPRING_ANNO_DIR = SPRING_ANNO_DIR

        total_frame_num = 0
        for idx, seq in enumerate(self.sequence_list):
            intrinsics = []
            with open(os.path.join(SPRING_DIR, f'{split}_cam_data', 'spring', split, seq, 'cam_data', 'intrinsics.txt'), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 4:
                        numbers = list(map(float, parts))
                        intrinsics.append(numbers)
            extrinsics = np.loadtxt(os.path.join(SPRING_DIR, f'{split}_cam_data', 'spring', split, seq, 'cam_data', 'extrinsics.txt'))
            # with open(os.path.join(SPRING_DIR, f'{split}_cam_data', 'spring', split, seq, 'cam_data', 'extrinsics.txt'), 'r') as f:
            #     for line in f:
            #         parts = line.strip().split()
            #         if len(parts) == 16:
            #             numbers = list(map(float, parts))
            #             extrinsics.append(numbers)
            for i, _ in enumerate(os.listdir(os.path.join(SPRING_ANNO_DIR, f'{split}_depth_left', seq))):
                self.cameras_map[idx].append({})
                extri = np.array(extrinsics[i]).reshape(4, 4)
                # extri = np.linalg.inv(extri)
                self.cameras_map[idx][-1]['extri'] = extri[:3, :]
                fx, fy, cx, cy = intrinsics[i]
                self.cameras_map[idx][-1]['intri'] = np.array([
                    [fx, 0,  cx],
                    [0,  fy, cy],
                    [0,  0,  1]
                ])

            total_frame_num += len(self.cameras_map[idx])

        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num
        
        # if self.debug:
        #     if self.training:
        #         self.len_train = self.sequence_list_len
        #     else:
        #         self.len_test = self.sequence_list_len
        status = "Training" if self.training else "Test"
        logging.info(f"{status}: Spring Data size: {self.sequence_list_len}")
        logging.info(f"{status}: Spring Data dataset length: {len(self)}")
        logging.info(f"{status}: Spring Data frame nums: {self.total_frame_num}")

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

            base_frame_path = os.path.join(self.SPRING_DIR, f'{self.split}_frame_left', 'spring', self.split, seq_name, f'frame_left')

            all_frames = sorted(os.listdir(base_frame_path))
            num_frames = len(all_frames)

            if ids is not None:
                start_idx = ids[0]
            elif img_per_seq is not None:
                while True:
                    if num_frames > img_per_seq:
                        break
                    seq_index = (seq_index + 1) % len(self.sequence_list)
                    seq_name = self.sequence_list[seq_index]
                    base_frame_path = os.path.join(self.SPRING_DIR, f'{self.split}_frame_left', 'spring', self.split, seq_name, 'frame_left')

                    all_frames = sorted(os.listdir(base_frame_path))
                    num_frames = len(all_frames)
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
                image_path = osp.join(base_frame_path, f'frame_left_{(global_idx + 1):04d}.png')
                image = read_image_cv2(image_path)

                depth_path = os.path.join(self.SPRING_ANNO_DIR, f'{self.split}_depth_left', seq_name, f'disp1_left_{(global_idx + 1):04d}_depth.png')
                depth_map = read_depth(depth_path, 1.0)

                depth_map = threshold_depth_map(depth_map, min_percentile=-1, max_percentile=80)
                depth_map[~np.isfinite(depth_map)] = 0 

                motion_path = osp.join(self.SPRING_DIR, f'{self.split}_maps', 'spring', self.split, seq_name, 'maps', 'rigidmap_FW_left', f'rigidmap_FW_left_{(global_idx + 1):04d}.png')
                if not os.path.exists(motion_path):
                    prefix, num_str = motion_path.rsplit('_', 1)
                    num_str = int(num_str[:-4]) - 1
                    motion_path = f"{prefix}_{num_str:04d}.png"

                motion = cv2.imread(motion_path, cv2.IMREAD_GRAYSCALE) > 128
                motion = motion.astype(np.int64)[::2, ::2]

                original_size = np.array(image.shape[:2])
                extri_opencv = np.array(self.cameras_map[seq_index][global_idx]["extri"])
                intri_opencv = np.array(self.cameras_map[seq_index][global_idx]["intri"])
                # cv2.imwrite('orig.png', image)

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

                # cv2.imwrite('processed.jpg', image)
                # depth_map_float16 = depth_map.astype(np.float16)
                # depth_uint16 = depth_map_float16.view(np.uint16)  # reinterpret bits
                # depth_uint16 = np.asarray(depth_uint16)  # ensure numpy array
                # img = Image.fromarray(depth_uint16, mode='I;16')
                # img.save('depth.png')
                # depth_map_float16 = motion.astype(np.float16) * 255
                # depth_uint16 = depth_map_float16.view(np.uint16)  # reinterpret bits
                # depth_uint16 = np.asarray(depth_uint16)  # ensure numpy array
                # img = Image.fromarray(depth_uint16, mode='I;16')
                # img.save('motion.png')
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

            set_name = "spring"

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
            # if not os.path.exists(f"colored_pointcloud_{seq_name}.ply"):
            #     pcd.export(f"colored_pointcloud_{seq_name}.ply")

            #     print(f"PLY 文件已保存为 colored_pointcloud_{seq_name}.ply")

            return batch
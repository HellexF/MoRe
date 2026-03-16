import gzip
import json
import os.path as osp
import os
import logging

import cv2
import random
import numpy as np


from data.dataset_util import *
from data.base_dataset import BaseDataset

class PointOdysseyDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        POINT_ODYSSEY_DIR: str = None,
        POINT_ODYSSEY_MOTION_DIR: str = None,
        min_num_images: int = 13,
        len_train: int = 5000,
        len_test: int = 200,
    ):
        """
        Initialize the SpringDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            POINT_ODYSSEY_DIR (str): Directory path to Poiny Odyssey data.
            POINT_ODYSSEY_MOTION_DIR (str): Directory path to Point Odyssey motion data.
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

        if POINT_ODYSSEY_DIR is None or POINT_ODYSSEY_MOTION_DIR is None:
            raise ValueError("Both POINT_ODYSSEY_DIR and POINT_ODYSSEY_MOTION_DIR must be specified.")

        if split == "train":
            split_name_list = ["train"]
            self.len_train = len_train
        elif split == "test":
            split_name_list = ["test"]
            split = "val"
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        split_path = os.path.join(POINT_ODYSSEY_DIR, split)
        all_entries = os.listdir(split_path)
        self.sequence_list = [entry for entry in all_entries if (os.path.isdir(os.path.join(split_path, entry)))]
        if self.debug:
            self.sequence_list = self.sequence_list[:2]
        self.split = split      
        
        self.seqlen = None
        self.cameras_map = [[] for _ in self.sequence_list]
        self.min_num_images = min_num_images
        self.seqlen = None

        logging.info(f"POINT_ODYSSEY_DIR is is {POINT_ODYSSEY_DIR}")
        self.POINT_ODYSSEY_DIR = POINT_ODYSSEY_DIR
        self.POINT_ODYSSEY_MOTION_DIR = POINT_ODYSSEY_MOTION_DIR

        total_frame_num = 0
        self.cam_info = {}
        for idx, seq in enumerate(self.sequence_list):
            anno_path = os.path.join(split_path, seq, 'anno.npz')
            anno = np.load(anno_path)
            intrinsics, extrinsics = anno['intrinsics'], anno['extrinsics']
            self.cam_info[seq] = {}
            self.cam_info[seq]['intrinsics'] = intrinsics
            self.cam_info[seq]['extrinsics'] = extrinsics
            total_frame_num += len(os.listdir(os.path.join(split_path, seq))) - 1

        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        # if self.debug:
        #     if self.training:
        #         self.len_train = self.sequence_list_len
        #     else:
        #         self.len_test = self.sequence_list_len
        status = "Training" if self.training else "Test"
        logging.info(f"{status}: Point Odyssey Data size: {self.sequence_list_len}")
        logging.info(f"{status}: Point Odyssey dataset length: {len(self)}")
        logging.info(f"{status}: Point Odyssey Data frame nums: {self.total_frame_num}")

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

            base_frame_path = os.path.join(self.POINT_ODYSSEY_DIR, self.split, seq_name, f'rgbs')
            all_frames = sorted(os.listdir(base_frame_path))[:-1]
            num_frames = len(all_frames)

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
                image_path = osp.join(base_frame_path, f'rgb_{global_idx:05d}.jpg')
                image = read_image_cv2(image_path)
                if image is None:
                    print(image_path)

                depth_path = osp.join(self.POINT_ODYSSEY_DIR, self.split, seq_name, 'depths', f'depth_{global_idx:05d}.png')
                depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                depth_map = depth_map.astype(np.float32) / 65535. * 1000.

                def safe_load_npy(path):
                    if not os.path.exists(path):
                        return None

                    try:
                        return np.load(path)
                    except EOFError:
                        return None

                motion_path = osp.join(self.POINT_ODYSSEY_MOTION_DIR, self.split, seq_name, 'masks', f'motion_mask_{global_idx:05d}.npy')
                motion = safe_load_npy(motion_path)
                if motion is None:
                    motion = np.zeros_like(depth_map)
                motion = motion.astype(np.int64)

                original_size = np.array(image.shape[:2])
                extri_opencv = self.cam_info[seq_name]['extrinsics'][global_idx][:3, :]
                intri_opencv = self.cam_info[seq_name]['intrinsics'][global_idx]

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

            set_name = "point_odyssey"

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
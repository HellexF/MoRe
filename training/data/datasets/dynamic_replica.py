import gzip
import json
import os.path as osp
import os
import logging

import cv2
import random
import numpy as np
import torch
import re

from data.dataset_util import *
from data.base_dataset import BaseDataset
import trimesh

class DynamicReplicaDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        DYNAMIC_REPLICA_DIR: str = None,
        min_num_images: int = 24    ,
        len_train: int = 5000,
        len_test: int = 48,
    ):
        """
        Initialize the SpringDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            DYNAMIC_REPLICA_DIR (str): Directory path to Dynamic Replica data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If DYNAMIC_REPLICA_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if DYNAMIC_REPLICA_DIR is None:
            raise ValueError("DYNAMIC_REPLICA_DIR must be specified.")

        if split == "train":
            split_name_list = ["train"]
            self.len_train = len_train
        elif split == "test":
            split = "valid"
            split_name_list = ["test"]
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}") 

        split_path = os.path.join(DYNAMIC_REPLICA_DIR, split)
        all_entries = os.listdir(split_path)
        self.sequence_list = [entry for entry in all_entries if (os.path.isdir(os.path.join(split_path, entry)))]
        if self.debug:
            self.sequence_list = self.sequence_list[:1]
        self.split = split   
        
        self.seqlen = None
        self.cameras_map = [[] for _ in self.sequence_list]
        self.min_num_images = min_num_images
        self.seqlen = None

        logging.info(f"DYNMIC_REPLICA_DIR is {DYNAMIC_REPLICA_DIR}")
        self.DYNAMIC_REPLICA_DIR = DYNAMIC_REPLICA_DIR

        total_frame_num = 0
        self.cam_info = {}
        with gzip.open(os.path.join(split_path, f'frame_annotations_{split}.jgz'), 'rt', encoding='utf-8') as f:
            annotations = json.load(f)
        for frame in annotations:
            name = f'{frame['sequence_name']}_source_{frame['camera_name']}'
            if not name in self.cam_info:
                new_dict = {
                    "extrinsics": [],
                    "intrinsics": []
                }
                self.cam_info[name] = new_dict
            
            width, height = frame['image']['size'][1], frame['image']['size'][0]

            focal_length_ndc = torch.tensor(frame['viewpoint']['focal_length'], dtype=torch.float)  # [fx_ndc, fy_ndc]
            principal_point_ndc = torch.tensor(frame['viewpoint']['principal_point'], dtype=torch.float)  # [cx_ndc, cy_ndc]
            format = frame['viewpoint']['intrinsics_format'].lower()

            half_image_size_wh = torch.tensor([width, height], dtype=torch.float) / 2.0

            if format == "ndc_norm_image_bounds":
                rescale = half_image_size_wh
            elif format == "ndc_isotropic":
                rescale = half_image_size_wh.min()
            else:
                raise ValueError(f"Unsupported intrinsics_format: {format}")

            focal_length_px = focal_length_ndc * rescale
            principal_point_px = half_image_size_wh - principal_point_ndc * rescale

            intrinsic = torch.tensor([
                [focal_length_px[0], 0, principal_point_px[0]],
                [0, focal_length_px[1], principal_point_px[1]],
                [0, 0, 1]
            ])

            R = np.array(frame['viewpoint']['R'])
            T = np.array(frame['viewpoint']['T'])
            T[..., :2] *= -1
            R[..., :, :2] *= -1

            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R.T
            extrinsic[:3, 3] = T

            self.cam_info[name]['intrinsics'].append(intrinsic)
            self.cam_info[name]['extrinsics'].append(extrinsic[:3, :])
            total_frame_num += 1

        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        # if self.eval:
        #     if self.training:
        #         self.len_train = self.sequence_list_len
        #     else:
        #         self.len_test = self.sequence_list_len
        status = "Training" if self.training else "Test"
        logging.info(f"{status}: Dynamic Replica Data size: {self.sequence_list_len}")
        logging.info(f"{status}: Dynamic Replica Data dataset length: {len(self)}")
        logging.info(f"{status}: Dynamic Replica Data frame nums: {self.total_frame_num}")

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

            base_frame_path = os.path.join(self.DYNAMIC_REPLICA_DIR, self.split, seq_name, f'images')

            def natural_key(s):
                return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

            all_frames = os.listdir(base_frame_path)
            all_frames = sorted([frame for frame in all_frames if frame.endswith('png')])
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
                image_path = osp.join(base_frame_path, f'{seq_name}-{global_idx:04d}.png')
                image = read_image_cv2(image_path)

                depth_path = os.path.join(self.DYNAMIC_REPLICA_DIR, self.split, seq_name, 'depths', f'{seq_name}_{global_idx:04d}.geometric.png')
                with Image.open(depth_path) as depth_pil:
                    depth_map = (
                        np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
                        .astype(np.float32)
                        .reshape((depth_pil.size[1], depth_pil.size[0]))
                    )

                motion_path = osp.join(self.DYNAMIC_REPLICA_DIR, self.split, seq_name, 'masks', f'{seq_name}_{global_idx:04d}.png')

                motion = cv2.imread(motion_path, cv2.IMREAD_GRAYSCALE) > 128
                motion = motion.astype(np.int64)

                original_size = np.array(image.shape[:2])
                extri_opencv = np.array(self.cam_info[seq_name]["extrinsics"][global_idx])
                intri_opencv = np.array(self.cam_info[seq_name]["intrinsics"][global_idx])

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

            set_name = "dynamic_replica"

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
            # if not os.path.exists(f"colored_pointcloud_{seq_name}.ply"):
            #     pcd.export(f"colored_pointcloud_{seq_name}.ply")

            #     print(f"PLY 文件已保存为 colored_pointcloud_{seq_name}.ply")

            return batch
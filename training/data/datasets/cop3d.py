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

class Cop3dDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        COP3D_DIR: str = None,
        min_num_images: int = 24    ,
        len_train: int = 5000,
        len_test: int = 1000,
    ):
        """
        Initialize the Cop3dDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            COP3D_DIR (str): Directory path to Dynamic Replica data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If COP3D_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if COP3D_DIR is None:
            raise ValueError("COP3D_DIR must be specified.")

        if split == "train":
            split_name_list = ["train"]
            self.len_train = len_train
        elif split == "test":
            split_name_list = ["test"]
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}") 

        self.sequence_list = []
        self.cam_info = {} 
        self.COP3D_DIR = COP3D_DIR
        categories = ["cat", "dog"]
        self.total_frame_num = 0
        for category in categories:
            category_dir = osp.join(COP3D_DIR, category)
            for seq in os.listdir(category_dir):
                if os.path.isdir(osp.join(category_dir, seq)):
                    self.sequence_list.append((category, seq))
                    self.total_frame_num += len(os.listdir(osp.join(category_dir, seq, "images")))
            # listfiles = os.listdir(osp.join(category_dir, "set_lists"))
            # subset_list_files = [f for f in listfiles if "manyview" in f]
            # if len(subset_list_files) <= 0:
            #     subset_list_files = [f for f in listfiles if "fewview" in f]

            # sequences_all = []
            # for subset_list_file in subset_list_files:
            #     with open(osp.join(category_dir, "set_lists", subset_list_file)) as f:
            #         subset_lists_data = json.load(f)
            #         sequences_all.extend(subset_lists_data[split])

            # sequences_numbers = sorted(set(seq_name for seq_name, _, _ in sequences_all))

            # # Load frame and sequence annotation files.
            # frame_file = osp.join(category_dir, "frame_annotations.jgz")
            # sequence_file = osp.join(category_dir, "sequence_annotations.jgz")

            # with gzip.open(frame_file, "r") as fin:
            #     frame_data = json.loads(fin.read())
            # with gzip.open(sequence_file, "r") as fin:
            #     sequence_data = json.loads(fin.read())

            # # Organize frame annotations per sequence.
            # frame_data_processed = {}
            # for f_data in frame_data:
            #     sequence_name = f_data["sequence_name"]
            #     frame_data_processed.setdefault(sequence_name, {})[
            #         f_data["frame_number"]
            #     ] = f_data

            # # Select sequences with quality above the threshold.
            # good_quality_sequences = set()
            # for seq_data in sequence_data:
            #     if seq_data["viewpoint_quality_score"] > min_quality:
            #         good_quality_sequences.add(seq_data["sequence_name"])
            # sequences_numbers = [
            #     seq_name for seq_name in sequences_numbers if seq_name in good_quality_sequences
            # ]
            # selected_sequences_numbers = sequences_numbers
            # selected_sequences_numbers_dict = {
            #     seq_name: [] for seq_name in selected_sequences_numbers
            # }

            # # Filter frames to only those from selected sequences.
            # sequences_all = [
            #     (seq_name, frame_number, filepath)
            #     for seq_name, frame_number, filepath in sequences_all
            #     if seq_name in selected_sequences_numbers_dict
            # ]

            # print(sequences_all)
        
        self.seqlen = None
        self.cameras_map = [[] for _ in self.sequence_list]
        self.min_num_images = min_num_images
        self.seqlen = None

        logging.info(f"COP3D_DIR is {COP3D_DIR}")

        self.sequence_list_len = len(self.sequence_list)

        # if self.eval:
        #     if self.training:
        #         self.len_train = self.sequence_list_len
        #     else:
        #         self.len_test = self.sequence_list_len
        status = "Training" if self.training else "Test"
        logging.info(f"{status}: Cop3D Data size: {self.sequence_list_len}")
        logging.info(f"{status}: Cop3D Data dataset length: {len(self)}")
        logging.info(f"{status}: Cop3D Data frame nums: {self.total_frame_num}")

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
            
            base_frame_path = os.path.join(self.COP3D_DIR, seq_name[0], seq_name[1], "images")
            num_frames = len(os.listdir(base_frame_path)) // 2

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
            filelist = sorted(os.listdir(base_frame_path))
            for global_idx in ids:
                image_path = osp.join(base_frame_path, filelist[2 * global_idx])
                image = read_image_cv2(image_path)

                depth_map = np.full(image.shape[:2], -1.0, dtype=float)
                motion_path = osp.join(base_frame_path.replace("images", "masks"), filelist[2 * global_idx].replace(".jpg", ".png"))
                if not osp.exists(motion_path):
                    motion = np.zeros_like(depth_map)
                else:
                    motion = cv2.imread(motion_path)
                    motion = np.any(motion[..., :3] != 0, axis=2).astype(np.float32) 

                original_size = np.array(image.shape[:2])
                cam_path = osp.join(base_frame_path, filelist[2 * global_idx + 1])
                with np.load(cam_path, allow_pickle=False) as data:
                    intri_opencv = data["camera_intrinsics"]
                    extri_opencv = data["camera_pose"][:3, :]
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

            set_name = "cop3d"

            batch = {
                "seq_name": set_name + "_" + seq_name[0] + "_" + seq_name[1],
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
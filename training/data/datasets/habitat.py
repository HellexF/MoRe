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
import h5py

from data.dataset_util import *
from data.base_dataset import BaseDataset
import trimesh
import pandas as pd

class HabitatDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        HABITAT_DIR: str = None,
        min_num_images: int = 24    ,
        len_train: int = 2000,
        len_test: int = 1000,
    ):
        """
        Initialize the HabitatDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            HABITAT_DIR (str): Directory path to habitat data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If WILDRGB_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.HABITAT_DIR = HABITAT_DIR

        if HABITAT_DIR is None:
            raise ValueError("HABITAT_DIR must be specified.")

        if split == "train":
            split_name_list = ["train"]
            self.len_train = len_train
        elif split == "test":
            split_name_list = ["test"]
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}") 

        self.sequence_list = []
        self.camera_intrinsics = {}
        self.frame_sizes = {}
        for seq in os.listdir(HYPERSIM_DIR):
            df_camera_parameters = pd.read_csv(all_metafile, index_col="scene_name")
            df_ = df_camera_parameters.loc[seq]

            width_pixels = int(df_["settings_output_img_width"])
            height_pixels = int(df_["settings_output_img_height"])

            proj_matrix = np.array(
                [
                    [df_["M_proj_00"], df_["M_proj_01"], df_["M_proj_02"], df_["M_proj_03"]],
                    [df_["M_proj_10"], df_["M_proj_11"], df_["M_proj_12"], df_["M_proj_13"]],
                    [df_["M_proj_20"], df_["M_proj_21"], df_["M_proj_22"], df_["M_proj_23"]],
                    [df_["M_proj_30"], df_["M_proj_31"], df_["M_proj_32"], df_["M_proj_33"]],
                ]
            )

            K00 = proj_matrix[0, 0] * width_pixels / 2.0
            K01 = -proj_matrix[0, 1] * width_pixels / 2.0
            K02 = (1.0 - proj_matrix[0, 2]) * width_pixels / 2.0
            K11 = proj_matrix[1, 1] * height_pixels / 2.0
            K12 = (1.0 + proj_matrix[1, 2]) * height_pixels / 2.0
            intrinsics =  np.array([[K00, K01, K02], [0.0, K11, K12], [0.0, 0.0, 1.0]])
            self.camera_intrinsics[seq] = intrinsics
            self.frame_sizes[seq] = [width_pixels, height_pixels]

            cam_dir = os.path.join(HYPERSIM_DIR, seq, "_detail", "metadata_cameras.csv")
            camera_ids = (
                pd.read_csv(
                    cam_dir,
                    header=None,
                    skiprows=1,
                )
                .to_numpy()
                .flatten()
            )
            for idx in camera_ids:
                if not os.path.isdir(os.path.join(HYPERSIM_DIR, seq, "images", f"scene_{idx}_final_hdf5")):
                    continue
                self.sequence_list.append(f"{seq}_{idx}")
        if split == "test":
            self.sequnce_list = self.sequence_list[-10:]
        if self.debug:
            self.sequence_list = self.sequence_list[:3]
        self.split = split   
        
        self.seqlen = None
        self.min_num_images = min_num_images

        logging.info(f"HYPERSIM_DIR is {HYPERSIM_DIR}")
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = 0
        for seq_name in self.sequence_list:
            _, p1, p2, cam = seq_name.split("_", 3)
            seq = f"ai_{p1}_{p2}"
            self.total_frame_num += len(os.listdir(os.path.join(self.HYPERSIM_DIR, seq, "images", f"scene_{cam}_final_hdf5"))) / 4

        status = "Training" if self.training else "Test"
        logging.info(f"{status}: HyperSim Data size: {self.sequence_list_len}")
        logging.info(f"{status}: HyperSim Data dataset length: {len(self)}")

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
            
            _, p1, p2, cam = seq_name.split("_", 3)
            seq = f"ai_{p1}_{p2}"
            image_root = os.path.join(self.HYPERSIM_DIR, seq, "images", f"scene_{cam}_final_hdf5")
            depth_root = os.path.join(self.HYPERSIM_DIR, seq, "images", f"scene_{cam}_geometry_hdf5")
            camera_root = os.path.join(self.HYPERSIM_DIR, seq, "_detail", cam)
            worldscale = (
                pd.read_csv(
                    os.path.join(self.HYPERSIM_DIR, seq, "_detail", "metadata_scene.csv"),
                    index_col="parameter_name",
                )
                .to_numpy()
                .flatten()[0]
                .astype(np.float32)
            )
            gamma = 1.0 / 2.2
            inv_gamma = 1.0 / gamma
            percentile = 90
            brightness_nth_percentile_desired = 0.8
            num_frames = len(os.listdir(image_root)) / 4
            print(num_frames)

            if ids is not None:
                start_idx = ids[0]
            elif img_per_seq is not None:
                if self.debug:
                    ids = np.arange(36)
                else:
                    ids = self.sample_fixed_interval_ids(num_frames, img_per_seq)
                start_idx = ids[0]
            else:
                start_idx = 0
                img_per_seq = num_frames
                ids = np.arange(num_frames)
                
            for global_idx in ids:
                _, global_idx, _ = os.listdir(image_root)[4 * global_idx].split(".", 2)
                image_path = osp.join(image_root, f"frame.{global_idx:04d}.color.hdf5")
                with h5py.File(image_path, "r") as f:
                    color = f["dataset"][:]

                depth_path = osp.join(depth_root, f"frame.{global_idx:04d}.depth_meters.hdf5")
                with h5py.File(depth_path, "r") as f:
                    distance = f["dataset"][:]

                camera_positions_hdf5_file = os.path.join(
                    camera_root, "camera_keyframe_positions.hdf5"
                )
                camera_orientations_hdf5_file = os.path.join(
                    camera_root, "camera_keyframe_orientations.hdf5"
                )

                with h5py.File(camera_positions_hdf5_file, "r") as f:
                    camera_positions = f["dataset"][global_idx]
                with h5py.File(camera_orientations_hdf5_file, "r") as f:
                    camera_orientations = f["dataset"][global_idx]
                R_cam2world = camera_orientations
                R_cam2world = R_cam2world @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                t_cam2world = camera_positions * worldscale
                T_cam2world = np.eye(4)
                T_cam2world[:3, :3] = R_cam2world
                T_cam2world[:3, 3] = t_cam2world
                extri_opencv = np.linalg.inv(T_cam2world)[:3, :]

                if not np.isfinite(T_cam2world).all():
                    print(f"frame_id={frame_id} T_cam2world is not finite.")
                    continue
                
                intri_opencv = self.camera_intrinsics[seq]
                width_pixels, height_pixels = self.frame_sizes[seq]
                focal = (intri_opencv[0, 0] + intri_opencv[1, 1]) / 2.0
                ImageplaneX = (
                    np.linspace(
                        (-0.5 * width_pixels) + 0.5,
                        (0.5 * width_pixels) - 0.5,
                        width_pixels,
                    )
                    .reshape(1, width_pixels)
                    .repeat(height_pixels, 0)
                    .astype(np.float32)[:, :, None]
                )
                ImageplaneY = (
                    np.linspace(
                        (-0.5 * height_pixels) + 0.5,
                        (0.5 * height_pixels) - 0.5,
                        height_pixels,
                    )
                    .reshape(height_pixels, 1)
                    .repeat(width_pixels, 1)
                    .astype(np.float32)[:, :, None]
                )
                ImageplaneZ = np.full([height_pixels, width_pixels, 1], focal, np.float32)
                Imageplane = np.concatenate([ImageplaneX, ImageplaneY, ImageplaneZ], axis=2)

                depth_map = distance / np.linalg.norm(Imageplane, axis=2) * focal

                render_entity = osp.join(depth_root, f"frame.{global_idx:04d}.render_entity_id.hdf5")
                with h5py.File(render_entity, "r") as f:
                    render_entity_id = f["dataset"][:].astype(np.int32)
                assert (render_entity_id != 0).all()
                valid_mask = render_entity_id != -1

                if np.sum(valid_mask) == 0:
                    scale = 1.0  # If there are no valid pixels, set scale to 1.0.
                else:
                    brightness = (
                        0.3 * color[:, :, 0] + 0.59 * color[:, :, 1] + 0.11 * color[:, :, 2]
                    )
                    brightness_valid = brightness[valid_mask]
                    eps = 0.0001  # Avoid division by zero.
                    brightness_nth_percentile_current = np.percentile(
                        brightness_valid, percentile
                    )
                    if brightness_nth_percentile_current < eps:
                        scale = 0.0
                    else:
                        scale = (
                            np.power(brightness_nth_percentile_desired, inv_gamma)
                            / brightness_nth_percentile_current
                        )

                color = np.power(np.maximum(scale * color, 0), gamma)
                image = (np.clip(color, 0.0, 1.0) * 255).astype(np.uint8)
                original_size = np.array(image.shape[:2])
                motion = np.zeros_like(depth_map)

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

            set_name = "hypersim"

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
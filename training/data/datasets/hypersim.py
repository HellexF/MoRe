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
import pandas as pd
import h5py

class HyperSimDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        HYPERSIM_DIR: str = None,
        HYPERSIM_ANNO_DIR: str = None,
        min_num_images: int = 24    ,
        len_train: int = 5000,
        len_test: int = 1000,
    ):
        """
        Initialize the HOI4DDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            HYPERSIM_DIR (str): Directory path to HyperSim data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If HYPERSIM_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if HYPERSIM_DIR is None:
            raise ValueError("HYPERSIM_DIR must be specified.")

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
        for seq in os.listdir(HYPERSIM_DIR):
            seq_dir = os.path.join(HYPERSIM_DIR, seq)
            camera_ids = (
                pd.read_csv(
                    os.path.join(seq_dir, "_detail", "metadata_cameras.csv"),
                    header=None,
                    skiprows=1,
                )
                .to_numpy()
                .flatten()
            )
            for camera_id in camera_ids:
                image_path = os.path.join(seq_dir, "images", f"scene_{camera_id}_final_hdf5")
                if os.path.isdir(image_path):
                    if len(os.listdir(image_path)) < 160:
                        continue
                    self.sequence_list.append((seq, camera_id))
                    all_metafile = os.path.join(HYPERSIM_ANNO_DIR, "metadata_camera_parameters.csv")
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
                    intrinsic = np.array([[K00, 0.0, K02], [0.0, K11, K12], [0.0, 0.0, 1.0]])
                    self.cam_info[f"{seq}_{camera_id}"] = {"intri": intrinsic, "width": width_pixels, "height": height_pixels}

        if split == "test":
            self.sequnce_list = self.sequence_list[-10:]
        if self.debug:
            self.sequence_list = self.sequence_list[:3]
        self.split = split   
        
        self.seqlen = None
        self.min_num_images = min_num_images
        self.seqlen = None

        logging.info(f"HYPERSIM_DIR is {HYPERSIM_DIR}")
        self.HYPERSIM_DIR = HYPERSIM_DIR

        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = 0
        for seq_name in self.sequence_list:
            self.total_frame_num += len(os.listdir(os.path.join(self.HYPERSIM_DIR, seq_name[0], "images", f"scene_{seq_name[1]}_final_hdf5"))) // 4

        # if self.eval:
        #     if self.training:
        #         self.len_train = self.sequence_list_len
        #     else:
        #         self.len_test = self.sequence_list_len
        status = "Training" if self.training else "Test"
        logging.info(f"{status}: HYPERSIM Data size: {self.sequence_list_len}")
        logging.info(f"{status}: HYPERSIM Data dataset length: {len(self)}")
        logging.info(f"{status}: HYPERSIM Data frame nums: {self.total_frame_num}")

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
            
            seq, cam = seq_name
            image_path = os.path.join(self.HYPERSIM_DIR, seq, "images", f"scene_{cam}_final_hdf5")
            depth_path = os.path.join(self.HYPERSIM_DIR, seq, "images", f"scene_{cam}_geometry_hdf5")
            cam_path = os.path.join(self.HYPERSIM_DIR, seq, "_detail", cam)
            frame_list = os.listdir(image_path)
            num_frames = len(frame_list) // 4
            gamma = 1.0 / 2.2  # Standard gamma correction exponent.
            inv_gamma = 1.0 / gamma
            percentile = 90  # Desired percentile brightness in the unmodified image.
            brightness_nth_percentile_desired = 0.8  # Desired brightness after scaling.


            if ids is not None:
                start_idx = ids[0]
            elif img_per_seq is not None:
                if self.debug:
                    ids = np.arange(12)
                else:
                    assert img_per_seq <= num_frames, f"{seq_name}_{img_per_seq}_{num_frames}"
                    ids = self.sample_fixed_interval_ids(num_frames, img_per_seq)
                start_idx = ids[0]
            else:
                start_idx = 0
                img_per_seq = num_frames
                ids = np.arange(num_frames)
                
            for global_idx in ids:
                global_idx = int(frame_list[global_idx // 4][6:10])

                worldscale = (
                    pd.read_csv(
                        os.path.join(self.HYPERSIM_DIR, seq, "_detail", "metadata_scene.csv"),
                        index_col="parameter_name",
                    )
                    .to_numpy()
                    .flatten()[0]
                    .astype(np.float32)
                )
                camera_positions_hdf5_file = os.path.join(
                    cam_path, "camera_keyframe_positions.hdf5"
                )
                camera_orientations_hdf5_file = os.path.join(
                    cam_path, "camera_keyframe_orientations.hdf5"
                )

                with h5py.File(camera_positions_hdf5_file, "r") as f:
                    camera_positions = f["dataset"][:]
                with h5py.File(camera_orientations_hdf5_file, "r") as f:
                    camera_orientations = f["dataset"][:]
                
                with h5py.File(os.path.join(image_path, f"frame.{global_idx:04d}.color.hdf5"), "r") as f:
                    color = f["dataset"][:]
                with h5py.File(os.path.join(depth_path, f"frame.{global_idx:04d}.depth_meters.hdf5"), "r") as f:
                    distance = f["dataset"][:]
                R_cam2world = camera_orientations[global_idx] 
                R_cam2world = R_cam2world @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                t_cam2world = camera_positions[global_idx] * worldscale
                T_cam2world = np.eye(4)
                T_cam2world[:3, :3] = R_cam2world
                T_cam2world[:3, 3] = t_cam2world
                extri_opencv = np.linalg.inv(T_cam2world)[:3, :]

                width_pixels = self.cam_info[f"{seq}_{cam}"]["width"]
                height_pixels = self.cam_info[f"{seq}_{cam}"]["height"]
                intri_opencv = self.cam_info[f"{seq}_{cam}"]["intri"]
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
                depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

                render_entity = os.path.join(depth_path, f"frame.{global_idx:04d}.render_entity_id.hdf5")
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
                image =  (np.clip(color, 0.0, 1.0) * 255).astype(np.uint8)
                original_size = np.array(image.shape[:2])
                motion = np.zeros_like(depth_map)

                # R = extri_opencv[:, :3]
                # t = extri_opencv[:, 3]

                # # 消除斜切并更新外参（保持 P 不变）
                # fx, s = intri_opencv[0, 0], intri_opencv[0, 1]
                # if abs(s) > 0 and abs(fx) > 1e-12:
                #     alpha = np.arctan2(s, fx)        # s = fx * tan(alpha)
                #     c, si = np.cos(alpha), np.sin(alpha)
                #     A = np.array([[c, -si, 0.0],
                #                 [si,  c, 0.0],
                #                 [0.0, 0.0, 1.0]], dtype=float)

                #     intri_opencv = intri_opencv @ A

                #     # 让新的焦距为正（必要时翻转列，同时保持投影等价）
                #     D = np.eye(3)
                #     if intri_opencv[0, 0] < 0: D[0, 0] = -1.0
                #     if intri_opencv[1, 1] < 0: D[1, 1] = -1.0
                #     intri_opencv = intri_opencv @ D
                #     A = A @ D

                #     intri_opencv = intri_opencv / intri_opencv[2, 2]

                #     R_new = A.T @ R
                #     t_new = A.T @ t
                # else:
                #     intri_opencv = intri_opencv.copy()
                #     R_new = R.copy()
                #     t_new = t.copy()

                # extri_opencv = np.hstack([R_new, t_new.reshape(3, 1)])

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

            set_name = "BlendedMVS"

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
            # seq_name = seq_name[0] + "_" + seq_name[1]
            # seq_name = seq_name.replace("/", "-")
            # if not os.path.exists(f"colored_pointcloud_{seq_name}.ply"):
            #     pcd.export(f"colored_pointcloud_{seq_name}.ply")

            #     print(f"PLY 文件已保存为 colored_pointcloud_{seq_name}.ply")

            return batch
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
import OpenEXR
import pandas as pd
# Enable OpenEXR support in OpenCV.
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Global constants
IMG_FORMAT = ".png"
rotate_flag = False
SENSOR_W = 36
SENSOR_H = 20.25
IMG_W = 1280
IMG_H = 720

# -----------------------------------------------------------------------------
# Helper functions for camera parameter conversion
# -----------------------------------------------------------------------------


def focalLength_mm2px(focalLength, dslr_sens, focalPoint):
    focal_pixel = (focalLength / dslr_sens) * focalPoint * 2
    return focal_pixel


def get_cam_int(fl, sens_w, sens_h, cx, cy):
    flx = focalLength_mm2px(fl, sens_w, cx)
    fly = focalLength_mm2px(fl, sens_h, cy)
    cam_mat = np.array([[flx, 0, cx], [0, fly, cy], [0, 0, 1]])
    return cam_mat


def unreal2cv2(points):
    # Permute coordinates: x --> y, y --> z, z --> x
    points = np.roll(points, 2, axis=1)
    # Invert the y-axis
    points = points * np.array([1.0, -1.0, 1.0])
    return points


def get_cam_trans(body_trans, cam_trans):
    cam_trans = np.array(cam_trans) / 100
    cam_trans = unreal2cv2(np.reshape(cam_trans, (1, 3)))
    body_trans = np.array(body_trans) / 100
    body_trans = unreal2cv2(np.reshape(body_trans, (1, 3)))
    trans = body_trans - cam_trans
    return trans


def get_cam_rotmat(pitch, yaw, roll):
    rotmat_yaw, _ = cv2.Rodrigues(np.array([[0, (yaw / 180) * np.pi, 0]], dtype=float))
    rotmat_pitch, _ = cv2.Rodrigues(np.array([pitch / 180 * np.pi, 0, 0]).reshape(3, 1))
    rotmat_roll, _ = cv2.Rodrigues(np.array([0, 0, roll / 180 * np.pi]).reshape(3, 1))
    final_rotmat = rotmat_roll @ (rotmat_pitch @ rotmat_yaw)
    return final_rotmat


def get_global_orient(cam_pitch, cam_yaw, cam_roll):
    pitch_rotmat, _ = cv2.Rodrigues(
        np.array([cam_pitch / 180 * np.pi, 0, 0]).reshape(3, 1)
    )
    roll_rotmat, _ = cv2.Rodrigues(
        np.array([0, 0, cam_roll / 180 * np.pi]).reshape(3, 1)
    )
    final_rotmat = roll_rotmat @ pitch_rotmat
    return final_rotmat


def convert_translation_to_opencv(x, y, z):
    t_cv = np.array([y, -z, x])
    return t_cv


def rotation_matrix_unreal(yaw, pitch, roll):
    yaw_rad = np.deg2rad(yaw)
    pitch_rad = np.deg2rad(pitch)
    roll_rad = np.deg2rad(roll)
    # Yaw (left-handed)
    R_yaw = np.array(
        [
            [np.cos(-yaw_rad), -np.sin(-yaw_rad), 0],
            [np.sin(-yaw_rad), np.cos(-yaw_rad), 0],
            [0, 0, 1],
        ]
    )
    # Pitch (right-handed)
    R_pitch = np.array(
        [
            [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
            [0, 1, 0],
            [-np.sin(pitch_rad), 0, np.cos(pitch_rad)],
        ]
    )
    # Roll (right-handed)
    R_roll = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0, np.sin(roll_rad), np.cos(roll_rad)],
        ]
    )
    R_unreal = R_roll @ R_pitch @ R_yaw
    return R_unreal


def convert_rotation_to_opencv(R_unreal):
    # Transformation matrix from Unreal to OpenCV coordinate system.
    C = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]])
    R_cv = C @ R_unreal @ C.T
    return R_cv


def get_rot_unreal(yaw, pitch, roll):
    yaw_rad = np.deg2rad(yaw)
    pitch_rad = np.deg2rad(pitch)
    roll_rad = np.deg2rad(roll)
    R_yaw = np.array(
        [
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1],
        ]
    )
    R_pitch = np.array(
        [
            [np.cos(pitch_rad), 0, -np.sin(pitch_rad)],
            [0, 1, 0],
            [np.sin(pitch_rad), 0, np.cos(pitch_rad)],
        ]
    )
    R_roll = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll_rad), np.sin(roll_rad)],
            [0, -np.sin(roll_rad), np.cos(roll_rad)],
        ]
    )
    R_unreal = R_yaw @ R_pitch @ R_roll
    return R_unreal


def get_extrinsics_unreal(R_unreal, t_unreal):
    cam_trans = np.array(t_unreal)
    ext = np.eye(4)
    ext[:3, :3] = R_unreal
    ext[:3, 3] = cam_trans.reshape(1, 3)
    return ext


def get_extrinsics_opencv(yaw, pitch, roll, x, y, z):
    R_unreal = get_rot_unreal(yaw, pitch, roll)
    t_unreal = np.array([x / 100.0, y / 100.0, z / 100.0])
    T_u2wu = get_extrinsics_unreal(R_unreal, t_unreal)
    T_opencv2unreal = np.array(
        [[0, 0, -1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    T_wu2ou = np.array(
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    return np.linalg.inv(T_opencv2unreal @ T_u2wu @ T_wu2ou)

class BedlamDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        BEDLAM_DIR: str = None,
        BEDLAM_ANNO_DIR: str = None,
        min_num_images: int = 13,
        len_train: int = 2000,
        len_test: int = 500,
    ):
        """
        Initialize the BedlamDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            BEDLAM_DIR (str): Directory path to Bedlam data.
            BEDLAM_ANNO_DIR (str): Directory path to processed Bedlam data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If BEDLAM_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if BEDLAM_DIR is None or BEDLAM_ANNO_DIR is None:
            raise ValueError("Both BEDLAM_DIR and BEDLAM_ANNO_DIR must be specified.")

        with open('/lpai/volumes/base-3da-ali-sh-mix/zhangwq/dataset/vggt/Bedlam/non_empty_scene_seq.txt', 'r') as f:
            self.sequence_list = [line.strip() for line in f if line.strip()]  
        if self.debug:
            self.sequence_list = self.sequence_list[:3]
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

        logging.info(f"BEDLAM_DIR is is {BEDLAM_DIR}")
        self.BEDLAM_DIR = BEDLAM_DIR
        self.BEDLAM_ANNO_DIR = BEDLAM_ANNO_DIR

        total_frame_num = 0

        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num
        
        status = "Training" if self.training else "Test"
        logging.info(f"{status}: Bedlam Data size: {self.sequence_list_len}")
        logging.info(f"{status}: Bedlam Data dataset length: {len(self)}")

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

            base_frame_path = os.path.join(self.BEDLAM_DIR, seq_name)
            base_depth_path = os.path.join(self.BEDLAM_DIR, seq_name.replace("png", "depth"))
            base_motion_path = os.path.join(self.BEDLAM_ANNO_DIR, seq_name.replace("png", "masks"))
            scene, _, seq = seq_name.split('/')
            cam_path = os.path.join(self.BEDLAM_ANNO_DIR, scene, "ground_truth", "camera", f"{seq}_camera.csv")
            cam_csv_data = pd.read_csv(cam_path)
            cam_csv_data = cam_csv_data.to_dict("list")
            cam_x = cam_csv_data["x"]
            cam_y = cam_csv_data["y"]
            cam_z = cam_csv_data["z"]
            cam_yaw_ = cam_csv_data["yaw"]
            cam_pitch_ = cam_csv_data["pitch"]
            cam_roll_ = cam_csv_data["roll"]
            fl = cam_csv_data["focal_length"]

            all_frames = sorted(os.listdir(base_frame_path))
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
                image_path = osp.join(base_frame_path, f'{seq}_{(global_idx):04d}.png')
                image = read_image_cv2(image_path)

                depth_path = os.path.join(base_depth_path, f'{seq}_{(global_idx):04d}_depth.exr')
                depth_map= OpenEXR.File(depth_path).parts[0].channels['Depth'].pixels
                depth_map = depth_map.astype(np.float32)/100.0
                depth_map[~np.isfinite(depth_map)] = 0.0
                depth_map = threshold_depth_map(depth_map, min_percentile=-1, max_percentile=95)
                # depth_map[depth_map > 80.0] = 0.0

                motion_path = os.path.join(base_motion_path, f'{seq}_{(global_idx):04d}_env.png')
                if not os.path.exists(motion_path):
                    motion = np.zeros_like(depth_map)
                else:
                    motion = cv2.imread(motion_path, cv2.IMREAD_GRAYSCALE) > 128
                    motion = 1 - motion.astype(np.int64)

                original_size = np.array(image.shape[:2])
                cam_pitch_ind = cam_pitch_[global_idx]
                cam_yaw_ind = cam_yaw_[global_idx]
                cam_roll_ind = cam_roll_[global_idx]

                intri_opencv = get_cam_int(fl[global_idx], SENSOR_W, SENSOR_H, IMG_W / 2.0, IMG_H / 2.0)

                rot_unreal = rotation_matrix_unreal(cam_yaw_ind, cam_pitch_ind, cam_roll_ind)
                rot_cv = convert_rotation_to_opencv(rot_unreal)
                trans_cv = convert_translation_to_opencv(
                    cam_x[global_idx] / 100.0, cam_y[global_idx] / 100.0, cam_z[global_idx] / 100.0
                )
                cam_ext_ = np.eye(4)
                cam_ext_[:3, :3] = rot_cv
                # The camera pose is computed as the inverse of the transformed translation.
                cam_ext_[:3, 3] = -rot_cv @ trans_cv

                extri_opencv = np.linalg.inv(cam_ext_)[:3, :]

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
            # seq_name = seq_name.replace('/', '-')
            # if not os.path.exists(f"colored_pointcloud_{seq_name}.ply"):
            #     pcd.export(f"colored_pointcloud_{seq_name}.ply")

            #     print(f"PLY 文件已保存为 colored_pointcloud_{seq_name}.ply")

            return batch
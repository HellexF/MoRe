import gzip
import json
import os.path as osp
import os
import logging
from torchvision import transforms as TF
import torch
import torch.nn.functional as F

import cv2
import random
import numpy as np
import trimesh

from training.data.dataset_util import *
from training.data.base_dataset import BaseDataset

class DycheckDataset(BaseDataset):
    def __init__(
        self,
        DYCHECK_DIR: str = None,
        type_: str = "video",
    ):

        logging.info(f"DYCHECK_DIR is {DYCHECK_DIR}")

        self.DYCHECK_DIR = DYCHECK_DIR
        # self.img_size = 518
        # self.patch_size = 14
        self.training = False
        self.landscape_check = False
        self.rescale = True
        self.rescale_aug = False
        self.type = type_

        self.sequence_list = os.listdir(self.DYCHECK_DIR)
        # self.sequence_list = ["mountain_1", "alley_1", "alley_2", "bamboo_1", "bamboo_2", "temple_2", "temple_3", "market_2", "market_5", "market_6", "cave_4", "ambush_4", "ambush_5", "ambush_6", "cave_2", "shaman_3", "sleeping_1", "sleeping_2"]
        if self.type =="mono":
            frame_list = []
            for sequence in self.sequence_list:
                for idx in range(len(os.listdir(os.path.join(self.DYCHECK_DIR, sequence, "rgb", "1x")))):
                    frame_list.append({"seq": sequence, "idx": idx})
            self.sequence_list =frame_list
        self.sequence_list_len = len(self.sequence_list)

        logging.info(f"DyCheck Data size: {self.sequence_list_len}")
    
    def __len__(self):
        return len(self.sequence_list)

    def __getitem__(self, idx_N) -> dict:
        images = []
        depths = []
        motions = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        motions = []
        original_sizes = []

        if self.type == "video":
            seq_name = self.sequence_list[idx_N]
            img_path = os.path.join(self.DYCHECK_DIR, seq_name, "rgb", "1x")

            for frame_num in range(len(os.listdir(img_path))):
                image_path = osp.join(img_path, f"0_{frame_num:05d}.png")
                image = read_image_cv2(image_path)
                original_size = np.array(image.shape[:2])

                depth_path = os.path.join(self.DYCHECK_DIR, seq_name, "depth", "1x", f"0_{frame_num:05d}.npy")
                depth_map = np.load(depth_path)

                target_image_shape = self.get_target_shape(original_size[0] / original_size[1])


                motion = np.zeros_like(depth_map, dtype=np.int64)

                camera_path = os.path.join(self.DYCHECK_DIR, seq_name, "camera", f"0_{frame_num:05d}.json")

                with open(camera_path, 'r') as f:
                    data = json.load(f)
    
                fx = fy = data["focal_length"]
                cx, cy = data["principal_point"]
                skew = data.get("skew", 0.0)
                pixel_aspect = data.get("pixel_aspect_ratio", 1.0)
                
                fy *= pixel_aspect

                intri_opencv = np.array([
                    [fx, skew, cx],
                    [0,  fy,  cy],
                    [0,   0,   1]
                ], dtype=np.float32)
                R = np.array(data["orientation"], dtype=np.float32)
                t = np.array(data["position"], dtype=np.float32).reshape(3, 1)

                w2c_R = R.T
                w2c_t = -R.T @ t
                w2c = np.eye(4, dtype=np.float32)
                w2c[:3, :3] = w2c_R
                w2c[:3, 3:] = w2c_t
                extri_opencv = w2c[:3, :]

                (
                    image,
                    depth_map,
                    extri_opencv,
                    intri_opencv,
                    world_coords_points,
                    cam_coords_points,
                    point_mask,
                    *_,
                    motion,
                ) = self.process_one_image(
                    image,
                    depth_map,
                    extri_opencv,
                    intri_opencv,
                    original_size,
                    target_image_shape,
                    filepath='',
                    motion=motion,
                )

                motion_path = image_path.replace("final", "masks")
                motion = Image.open(motion_path)

                # If there's an alpha channel, blend onto white background:
                if motion.mode == "RGBA":
                    # Create white background
                    background = Image.new("RGBA", motion.size, (255, 255, 255, 255))
                    # Alpha composite onto the white background
                    motion = Image.alpha_composite(background, motion)

                # 转换为灰度图（白色填充透明区域）
                motion = motion.convert("L")
                to_tensor = TF.ToTensor()
                motion = to_tensor(motion)
                # patch_size = 14
                patch_scores = torch.log(1 - F.avg_pool2d(motion, kernel_size=self.patch_size, stride=self.patch_size))
                patch_scores = patch_scores.flatten(1).cpu().numpy()

                images.append(image)
                depths.append(depth_map)
                motions.append(patch_scores)
                extrinsics.append(extri_opencv)
                intrinsics.append(intri_opencv)
                cam_points.append(cam_coords_points)
                world_points.append(world_coords_points)
                point_masks.append(point_mask)
                image_paths.append(image_path)
                original_sizes.append(original_size)
            ids = range(len(os.listdir(img_path)))

        else:
            seq_name, idx = self.sequence_list[idx_N]["seq"], self.sequence_list[idx_N]["idx"]
            image_path = osp.join(img_path, f"0_{idx:05d}.png")
            image = read_image_cv2(image_path)
            original_size = np.array(image.shape[:2])

            depth_path = os.path.join(self.DYCHECK_DIR, seq_name, "depth", "1x", f"0_{idx:05d}.npy")
            depth_map = np.load(depth_path)

            target_image_shape = self.get_target_shape(original_size[0] / original_size[1])


            motion = np.zeros_like(depth_map, dtype=np.int64)

            camera_path = os.path.join(self.DYCHECK_DIR, seq_name, "camera", f"0_{idx:05d}.json")

            with open(json_path, 'r') as f:
                data = json.load(f)

            fx = fy = data["focal_length"]
            cx, cy = data["principal_point"]
            skew = data.get("skew", 0.0)
            pixel_aspect = data.get("pixel_aspect_ratio", 1.0)
            
            fy *= pixel_aspect

            intri_opencv = np.array([
                [fx, skew, cx],
                [0,  fy,  cy],
                [0,   0,   1]
            ], dtype=np.float32)
            R = np.array(data["orientation"], dtype=np.float32)
            t = np.array(data["position"], dtype=np.float32).reshape(3, 1)

            w2c_R = R.T
            w2c_t = -R.T @ t
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = w2c_R
            w2c[:3, 3:] = w2c_t
            extri_opencv = w2c[:3, :]

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                *_,
                motion,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath='',
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
            ids = [idx]

        set_name = "dycheck"
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

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

class SintelDataset(BaseDataset):
    def __init__(
        self,
        SINTEL_IMG_DIR: str = None,
        SINTEL_ANNOTATION_DIR: str = None,
        type_: str = "video",
    ):

        logging.info(f"SINTEL_DIR is {SINTEL_IMG_DIR}")

        self.SINTEL_IMG_DIR = SINTEL_IMG_DIR
        self.SINTEL_ANNOTATION_DIR = SINTEL_ANNOTATION_DIR
        # self.img_size = 518
        # self.patch_size = 14
        self.training = False
        self.landscape_check = False
        self.rescale = True
        self.rescale_aug = False
        self.type = type_

        self.sequence_list = os.listdir(self.SINTEL_IMG_DIR)
        # self.sequence_list = ["mountain_1", "alley_1", "alley_2", "bamboo_1", "bamboo_2", "temple_2", "temple_3", "market_2", "market_5", "market_6", "cave_4", "ambush_4", "ambush_5", "ambush_6", "cave_2", "shaman_3", "sleeping_1", "sleeping_2"]
        if self.type =="mono":
            frame_list = []
            for sequence in self.sequence_list:
                for idx in range(len(os.listdir(os.path.join(self.SINTEL_IMG_DIR, sequence)))):
                    frame_list.append({"seq": sequence, "idx": idx})
            self.sequence_list =frame_list
        self.sequence_list_len = len(self.sequence_list)

        logging.info(f"Sintel Data size: {self.sequence_list_len}")
    
    def __len__(self):
        return len(self.sequence_list)

    def __getitem__(self, idx_N) -> dict:
        def depth_read(filename):
            """ Read depth data from file, return as numpy array. """
            f = open(filename,'rb')
            check = np.fromfile(f,dtype=np.float32,count=1)[0]
            # assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
            width = np.fromfile(f,dtype=np.int32,count=1)[0]
            height = np.fromfile(f,dtype=np.int32,count=1)[0]
            size = width*height
            assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
            depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
            return depth
        def cam_read(filename):
            """ Read camera data, return (M,N) tuple.
            
            M is the intrinsic matrix, N is the extrinsic matrix, so that

            x = M*N*X,
            with x being a point in homogeneous image pixel coordinates, X being a
            point in homogeneous world coordinates.
            """
            f = open(filename,'rb')
            check = np.fromfile(f,dtype=np.float32,count=1)[0]
            # assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
            M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
            N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
            return M,N
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
            img_path = os.path.join(self.SINTEL_IMG_DIR, seq_name)

            for frame_num in range(len(os.listdir(img_path))):
                image_path = osp.join(img_path, f"frame_{(frame_num + 1):04d}.png")
                image = read_image_cv2(image_path)
                original_size = np.array(image.shape[:2])

                depth_path = os.path.join(self.SINTEL_ANNOTATION_DIR, 'depth', seq_name, f"frame_{(frame_num + 1):04d}.dpt")
                depth_map = depth_read(depth_path)

                target_image_shape = self.get_target_shape(original_size[0] / original_size[1])


                motion = np.zeros_like(depth_map, dtype=np.int64)

                camera_path = os.path.join(self.SINTEL_ANNOTATION_DIR, 'camdata_left', seq_name, f"frame_{(frame_num + 1):04d}.cam")
                intri_opencv, extri_opencv = cam_read(camera_path)
                # T_bl_w2c = np.eye(4, 4)
                # T_bl_w2c[:3, :] = extri_opencv

                # S = np.diag([1, -1, -1, 1]) 

                # T_cv_w2c = S @ T_bl_w2c

                # R_cv = T_cv_w2c[:3, :3]
                # t_cv = T_cv_w2c[:3, 3]
                
                # extri_opencv = np.hstack([R_cv, t_cv.reshape(3, 1)])
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
            image_path = osp.join(self.SINTEL_IMG_DIR, seq_name, f"frame_{(idx + 1):04d}.png")
            image = read_image_cv2(image_path)
            original_size = np.array(image.shape[:2])

            depth_path = os.path.join(self.SINTEL_ANNOTATION_DIR, 'depth', seq_name, f"frame_{(idx + 1):04d}.dpt")
            depth_map = depth_read(depth_path)

            target_image_shape = self.get_target_shape(original_size[0] / original_size[1])

            motion = np.zeros_like(depth_map, dtype=np.int64)
            camera_path = os.path.join(self.SINTEL_ANNOTATION_DIR, 'camdata_left', seq_name, f"frame_{(idx + 1):04d}.cam")
            intri_opencv, extri_opencv = cam_read(camera_path)
            R_b2cv = np.array([
                [1,  0,  0],
                [0, -1,  0],
                [0,  0, -1]
            ])

            R_cw = extri_opencv[:3, :3]
            t_cw = extri_opencv[:3, 3]

            R_wc = R_cw.T
            t_wc = -R_wc @ t_cw
            
            extri_opencv = np.hstack([R_wc, t_wc.reshape(3, 1)])

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

        set_name = "sintel"
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

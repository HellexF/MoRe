import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core import calibration
from projectaria_tools.projects.adt import (
   AriaDigitalTwinDataProvider,
   AriaDigitalTwinSkeletonProvider,
   AriaDigitalTwinDataPathsProvider,
   bbox3d_to_line_coordinates,
   bbox2d_to_image_coordinates,
   utils as adt_utils,
)


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,  2*qx*qy - 2*qz*qw,   2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,  1 - 2*qx*qx - 2*qz*qz,   2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,  2*qy*qz + 2*qx*qw,   1 - 2*qx*qx - 2*qy*qy]
    ], dtype=np.float32)


def get_camera_K(cam_calib):
    """尝试新版 get_projection_matrix，不行则 fallback"""
    try:
        return np.array(cam_calib.get_projection_matrix(), dtype=np.float32)
    except AttributeError:
        cx, cy = cam_calib.get_principal_point()
        fx, fy = cam_calib.get_focal_lengths()
        return np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]], dtype=np.float32)


def load_adt_frames(dataset_dir, save_dir=None):
    dataset_dir = Path(dataset_dir)
    if save_dir:
        save_dir = Path(save_dir)
        (save_dir / "rgb").mkdir(parents=True, exist_ok=True)
        (save_dir / "depth").mkdir(parents=True, exist_ok=True)
        (save_dir / "mask").mkdir(parents=True, exist_ok=True)
        (save_dir / "pose").mkdir(parents=True, exist_ok=True)

    # -------- RGB provider --------
    rgb_provider = data_provider.create_vrs_data_provider(str(dataset_dir / "synthetic_video.vrs"))
    rgb_streams = rgb_provider.get_all_streams()
    print("可用 RGB 流：")
    for s in rgb_streams:
        print(" ", str(s))
    rgb_stream_id = rgb_streams[0]  # 214-1 / 1201-1 / 1201-2

    calib = rgb_provider.get_device_calibration()
    rgb_label = str(rgb_stream_id)  # 先直接用 ID string

    STREAM_ID_TO_LABEL = {
        "214-1": "camera-rgb",
        "1201-1": "camera-slam-left",
        "1201-2": "camera-slam-right"
    }
    rgb_label = STREAM_ID_TO_LABEL.get(rgb_label, rgb_label)
    cam_calib = calib.get_camera_calib(rgb_label)

    calib = rgb_provider.get_device_calibration()
    cam_calib = calib.get_camera_calib(rgb_label)  # 传 label 字符串
    K_mat = get_camera_K(cam_calib)
    if save_dir:
        np.save(save_dir / "K.npy", K_mat)

    # -------- Depth provider --------
    depth_provider = data_provider.create_vrs_data_provider(str(dataset_dir / "depth_images.vrs"))
    depth_streams = depth_provider.get_all_streams()
    print("可用 Depth 流：")
    for s in depth_streams:
        print(" ", str(s))
    depth_stream_id = depth_streams[0]

    # -------- Segmentation provider --------
    seg_provider = data_provider.create_vrs_data_provider(str(dataset_dir / "segmentations.vrs"))
    seg_streams = seg_provider.get_all_streams()
    print("可用 Segmentation 流：")
    for s in seg_streams:
        print(" ", str(s))
    seg_stream_id = seg_streams[0]

    # -------- 轨迹 --------
    traj_df = pd.read_csv(dataset_dir / "aria_trajectory.csv")

    # -------- 动态物体ID --------
    with open(dataset_dir / "instances.json", "r") as f:
        instances_info = json.load(f)
    moving_ids = [int(k) for k, v in instances_info.items()
                  if v.get("motion", False) or v.get("motion_type", "").lower() in ["dynamic"]]
    print("运动物体 IDs:", moving_ids)

    paths_provider = AriaDigitalTwinDataPathsProvider(str(dataset_dir))
    data_paths = paths_provider.get_datapaths()
    print(data_paths)
    gt_provider = AriaDigitalTwinDataProvider(data_paths)

    stream_id = StreamId("214-1")
    depth_stream_id = StreamId("345-1")
    img_timestamps_ns = gt_provider.get_aria_device_capture_timestamps_ns(stream_id)
    print(img_timestamps_ns)
    num_frames = len(img_timestamps_ns)
    print(f"共 {num_frames} 帧, 开始提取...")

    for i, ts_ns in enumerate(img_timestamps_ns):
        image_with_dt = gt_provider.get_aria_image_by_timestamp_ns(ts_ns, stream_id)
        rgb_img = image_with_dt.data().to_numpy_array()
        print(rgb_img.shape, ts_ns)

        depth_with_dt = gt_provider.get_depth_image_by_timestamp_ns(ts_ns, depth_stream_id)

        # draw image
        depth_m = depth_with_dt.data().to_numpy_array() / 1e3

        # 位姿
        aria3dpose_with_dt = gt_provider.get_aria_3d_pose_by_timestamp_ns(ts_ns)
        aria3dpose = aria3dpose_with_dt.data()

        # 相机→设备
        transform_device_camera = gt_provider.get_aria_transform_device_camera(stream_id)
        # 相机→世界 = (相机→设备) × (设备→世界)
        T_world_cam = transform_device_camera.to_matrix() @ aria3dpose.transform_scene_device.inverse().to_matrix()

        # 分割
        seg_sensor_data = seg_provider.get_sensor_data_by_index(seg_stream_id, i)
        seg_img = seg_sensor_data.image_data_and_record.image()  # numpy array
        motion_mask = np.isin(seg_img, moving_ids).astype(np.uint8) * 255

        # 保存
        cv2.imwrite(str(save_dir / "rgb" / f"{i:05d}.png"), rgb_img)
        cv2.imwrite(str(save_dir / "depth" / f"{i:05d}.png"), depth_m)
        cv2.imwrite(str(save_dir / "mask" / f"{i:05d}.png"), motion_mask)
        np.save(save_dir / "pose" / f"{i:05d}.npy", T_world_cam)


if __name__ == "__main__":
    dataset_dir = "/lpai/dataset/adt/0-1-0/datasets/Apartment_release_clean_seq131_M1292"  # 你的ADT目录
    save_dir = "/lpai/volumes/base-3da-ali-sh-mix/zhangwq/dataset/vggt/ADT/Apartment_release_clean_seq131_M1292"

    frames = load_adt_frames(dataset_dir, save_dir)
    print(f"共提取到 {len(frames)} 帧数据")
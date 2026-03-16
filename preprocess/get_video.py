import cv2
import os
from pathlib import Path

def frames_to_video(image_folder, output_video):
    # 参数配置
    fps = 30  # 帧率

    # 获取图片文件列表（按文件名排序）
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))])

    # 确保至少有一张图
    if not images:
        raise ValueError("没有找到图片文件")

    # 读取第一张图确定尺寸
    first_frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = first_frame.shape

    # 设置视频编码器和输出对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 也可以试试 'avc1'，具体取决于系统支持
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # 写入所有图片帧
    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"跳过无效图像: {img_name}")
            continue
        out.write(frame)

    out.release()
    print(f"视频已保存为 {output_video}")

dataset_path = '/lpai/dataset/action-model-group-data-weight/0-3-0/TartanAir/dataset/rgb'
for scene in os.listdir(dataset_path):
    for dif in os.listdir(os.path.join(dataset_path, scene)):
        for seq in os.listdir(os.path.join(dataset_path, scene, dif)):
            image_path = os.path.join(dataset_path, scene, dif, seq, 'image_left')
            output_path = os.path.join('results', 'tartan', f'{scene}_{dif}_{seq}.mp4')
            frames_to_video(image_path, output_path)

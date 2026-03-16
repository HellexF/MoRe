from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os


from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def save_masks_png(masks, save_path, image=None):
    """
    masks: list of dicts, each with 'segmentation'
    save_path: "xxx.png"
    image: optional background (H, W, 3)
    """
    H, W = masks[0]['segmentation'].shape

    if image is None:
        vis = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        vis = image.copy()

    for mask in masks:
        seg = mask['segmentation']
        if seg.dtype != np.bool_:
            seg = seg > 0

        color = np.random.randint(0, 255, size=3, dtype=np.uint8)

        vis[seg] = (0.5 * vis[seg] + 0.5 * color).astype(np.uint8)

    plt.figure(figsize=(10, 10))
    plt.imshow(vis)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"Saved to {save_path}")

sam2_checkpoint = ""
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = 'cuda'
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
mask_generator_2 = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=16,   
    points_per_batch=128,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.6,
    stability_score_offset=0.7,
    crop_n_layers=1,
    box_nms_thresh=0.95,  
    crop_n_points_downscale_factor=2,
    min_mask_region_area=200.0, 
    use_m2m=True,
)

video_base_dir = ""
base_mask_dir = ""

output_base_dir = '../../eval_results/motion_masks/DAVIS'

for seq in os.listdir(base_mask_dir):
    print(seq)
    video_dir = os.path.join (video_base_dir, seq)
    mask_dir = os.path.join(base_mask_dir, seq, "masks")
    output_dir = os.path.join(output_base_dir, seq)
    os.makedirs(output_dir, exist_ok=True)
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    for i in range(len(frame_names)):
        w, h = Image.open(os.path.join(video_dir, frame_names[i])).size
        # inference_state = predictor.init_state(video_path=video_dir)

        ann_obj_id = 1


        mask_path = os.path.join(mask_dir, f"mask_{i:05d}.png")
        mask = np.array(Image.open(mask_path))  # shape = (H, W)
        # w, h = mask.shape

        mask_up = cv2.resize(mask / 255, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_up_norm = np.clip(mask_up, 0, 1)
        mask_up_uint8 = (mask_up_norm * 255).astype(np.uint8)

        mask_flat = mask_up.flatten()
        k = int(len(mask_flat) * 0.01)
        if k < 1:
            k = 1

        threshold = np.partition(mask_flat, -k)[-k]

        k = int(len(mask_flat) * 0.05)
        if k < 1:
            k = 1

        looser_threshold = np.partition(mask_flat, -k)[-k]
        cv2.imwrite(os.path.join(output_dir, f"mask_orig_{i:05d}.png"), (mask_up >= looser_threshold).astype(np.uint8) * 255)

        ys, xs = np.where(mask_up >= threshold)

        points = np.stack([xs, ys], axis=1).astype(np.float32)

        num_points = 20

        grid_size = int(np.sqrt((h*w) / num_points))

        points_list = []

        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                y2 = min(y + grid_size, h)
                x2 = min(x + grid_size, w)

                ys_, xs_ = np.where(mask_up[y:y2, x:x2] >= threshold)

                if len(xs_) > 0:
                    idx = np.random.randint(len(xs_))
                    points_list.append([int(x + xs_[idx]), int(y + ys_[idx])])

        coarse_points = np.array(points_list, dtype=np.float32)

        dense_list = []

        densify_radius = 1000 
        densify_k = 10          

        for px, py in coarse_points:

            y0 = int(max(py - densify_radius, 0))
            y1 = int(min(py + densify_radius + 1, h))
            x0 = int(max(px - densify_radius, 0))
            x1 = int(min(px + densify_radius + 1, w))

            ys2, xs2 = np.where(mask_up[y0:y1, x0:x1] >= threshold)

            if len(xs2) > 0:
                k2 = min(densify_k, len(xs2))
                idxs2 = np.random.choice(len(xs2), size=k2, replace=False)

                for idx in idxs2:
                    dense_list.append([int(x0 + xs2[idx]), int(y0 + ys2[idx])])

        points = np.array(dense_list, dtype=np.float32)

        min_dist = 1000

        filtered = []
        for p in points:
            px, py = p
            keep = True

            for qx, qy in filtered:
                if (px - qx) ** 2 + (py - qy) ** 2 < min_dist ** 2:
                    keep = False
                    break

            if keep:
                filtered.append([px, py])

        points = np.array(filtered[:1], dtype=np.float32)

        vis = cv2.cvtColor(mask_up_uint8, cv2.COLOR_GRAY2BGR)

        for (x, y) in points.astype(int):
            cv2.circle(vis, (int(x), int(y)), radius=2, color=(0, 0, 255), thickness=-1)

        # for (x, y) in coarse_points.astype(int):
        #     cv2.circle(vis, (int(x), int(y)), radius=2, color=(0, 255, 0), thickness=-1)

        cv2.imwrite("points_on_mask_up.png", vis)

        image = Image.open(os.path.join(video_dir, frame_names[i]))
        image = np.array(image.convert("RGB"))
        # image = cv2.resize(image, (h, w), interpolation=cv2.INTER_LINEAR)
        masks = mask_generator_2.generate(image)
        save_masks_png(masks, f"mask_seg_{i:05d}.png")
        # cv2.imwrite(os.path.join(output_dir, f"mask_{i:05d}.png"), final_mask_png)
import os
import time
import argparse
import numpy as np
import torch
import cv2
import trimesh
import matplotlib
import onnxruntime
from PIL import Image
from scipy.spatial.transform import Rotation
from hydra import initialize, compose
from hydra.utils import instantiate

# Import custom utilities (ensure these are in your PYTHONPATH)
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from visual_util import segment_sky, apply_scene_alignment

class MoReInference:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        
        # Auto-detect precision support: bfloat16 for Ampere+, float16 for others
        if self.device.type == 'cuda' and torch.cuda.get_device_capability()[0] >= 8:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16
            
        self.model = self._load_model()
        self.skyseg_session = None

    def _load_model(self):
        print(f"[*] Loading model from {self.args.ckpt_path}...")
        
        # Split path for Hydra initialization
        config_dir = os.path.dirname(self.args.config_path)
        config_name = os.path.basename(self.args.config_path).replace('.yaml', '')
        
        with initialize(version_base=None, config_path=config_dir):
            cfg = compose(config_name=config_name)
        
        model = instantiate(cfg.model, _recursive_=False)
        checkpoint = torch.load(self.args.ckpt_path, map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        
        return model.to(self.device).eval()

    def _get_opengl_matrix(self) -> np.ndarray:
        """Returns the 4x4 matrix to convert to OpenGL coordinate system."""
        matrix = np.identity(4)
        matrix[1, 1], matrix[2, 2] = -1, -1
        return matrix

    def save_ply(self, filename, points, colors=None):
        """Standard PLY exporter for point clouds."""
        points = points.reshape(-1, 3)
        with open(filename, 'w') as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {points.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            if colors is not None:
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for i in range(len(points)):
                line = f"{points[i][0]} {points[i][1]} {points[i][2]}"
                if colors is not None:
                    line += f" {colors[i][0]} {colors[i][1]} {colors[i][2]}"
                f.write(line + "\n")

    def run(self):
        # 1. Setup directories
        os.makedirs(self.args.output_dir, exist_ok=True)
        glb_dir = os.path.join(self.args.output_dir, 'glb')
        os.makedirs(os.path.join(glb_dir, "sky_masks"), exist_ok=True)

        # 2. Load and preprocess images
        img_paths = sorted([os.path.join(self.args.image_path, f) 
                           for f in os.listdir(self.args.image_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not img_paths:
            raise FileNotFoundError(f"No images found in {self.args.image_path}")
            
        print(f"[*] Processing {len(img_paths)} images on {self.device}...")
        images = load_and_preprocess_images(img_paths).to(self.device)
        images_input = images[None] # Add batch dimension (1, S, C, H, W)

        # 3. Model Inference
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                start_time = time.time()
                tokens, ps_idx, _ = self.model.aggregator(images_input)
                
                # Predict Cameras (Extrinsics/Intrinsics)
                pose_enc = self.model.camera_head(tokens, first_token=True)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_input.shape[-2:])
                
                # Predict Depth and Unproject to 3D Points
                depth_map, depth_conf = self.model.depth_head(tokens, images_input, ps_idx)
                point_map_unproj = unproject_depth_map_to_point_map(
                    depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
                )
                
                motion_mask = None
                if self.args.predict_motion:
                    motion_mask = self.model.motion_head(tokens, images_input, ps_idx)
                
                print(f"[!] Inference completed in {time.time() - start_time:.2f}s")

        # 4. Sky Segmentation & Confidence Masking
        H, W = images.shape[-2:]
        sky_masks = []
        for i, path in enumerate(img_paths):
            mask_path = os.path.join(glb_dir, "sky_masks", f'{i}.png')
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                if self.skyseg_session is None:
                    self.skyseg_session = onnxruntime.InferenceSession(self.args.skyseg_onnx)
                mask = segment_sky(path, self.skyseg_session, mask_path)
            
            if mask.shape[:2] != (H, W):
                mask = cv2.resize(mask, (W, H))
            sky_masks.append(mask)
        
        # Binary mask for sky (1.0 for non-sky, 0.0 for sky)
        sky_mask_array = np.array(sky_masks) / 255.0
        depth_conf_np = depth_conf.squeeze(0).cpu().float().numpy() * (sky_mask_array > 0.1)

        # 5. Export results
        self._export_results(img_paths, images, point_map_unproj, depth_map, 
                             depth_conf_np, extrinsic, intrinsic, motion_mask)

    def _export_results(self, img_paths, images, point_map_unproj, depth_map, 
                        depth_conf_np, extrinsic, intrinsic, motion_mask):
        print("[*] Exporting files...")
        
        # Calculate confidence threshold based on percentile
        conf_flat = depth_conf_np.flatten()
        conf_threshold = np.percentile(conf_flat, self.args.conf_thres) if self.args.conf_thres > 0 else 0.0

        # Create output sub-directories
        for folder in ["masks", "depth", "conf", "extrinsic", "intrinsic"]:
            os.makedirs(os.path.join(self.args.output_dir, folder), exist_ok=True)

        ext_np = extrinsic.squeeze(0).cpu().numpy()
        int_np = intrinsic.squeeze(0).cpu().numpy()
        
        # Process per frame
        for i in range(len(img_paths)):
            # Save Matrices
            np.save(f"{self.args.output_dir}/extrinsic/{i:05d}.npy", ext_np[i])
            np.save(f"{self.args.output_dir}/intrinsic/{i:05d}.npy", int_np[i])

            # Save Depth Data and Visualizations
            d_map = depth_map[0, i].cpu().float().numpy()
            np.save(f"{self.args.output_dir}/depth/{i:05d}.npy", d_map)
            d_vis = (d_map / (np.max(d_map) + 1e-6) * 255).astype(np.uint8)
            cv2.imwrite(f"{self.args.output_dir}/depth/depth_vis_{i:05d}.png", d_vis)

            # Filter Points by Confidence and Save PLY
            pts = point_map_unproj[i].reshape(-1, 3).cpu().float().numpy()
            conf = depth_conf_np[i].reshape(-1)
            valid_mask = (conf >= conf_threshold) & (conf > 1e-5)
            
            # Extract RGB colors
            rgb = (images[i].permute(1, 2, 0).cpu().numpy() * 255).reshape(-1, 3).astype(np.uint8)
            self.save_ply(f"{self.args.output_dir}/frame_{i:05d}.ply", pts[valid_mask], rgb[valid_mask])

            # Save Motion Masks
            if motion_mask is not None:
                m_mask = torch.sigmoid(motion_mask[0, i, ..., 0]).cpu().float().numpy()
                Image.fromarray((m_mask * 255).astype(np.uint8)).save(
                    f"{self.args.output_dir}/masks/mask_{i:05d}.png"
                )

        # Optional: Save Global Scene (GLB)
        scene = trimesh.Scene()
        # Integration of alignment and camera visualization could go here
        scene.export(os.path.join(self.args.output_dir, 'glb', 'scene.glb'))
        
        print(f"[!] Processing finished. Outputs located in: {self.args.output_dir}")

def get_args():
    parser = argparse.ArgumentParser(description="MoRe Inference Pipeline")
    
    # Path arguments
    parser.add_argument("--config_path", type=str, required=True, help="Hydra config file path")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Model checkpoint path (.pth)")
    parser.add_argument("--image_path", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory for all outputs")
    parser.add_argument("--skyseg_onnx", type=str, default="skyseg.onnx", help="Path to sky segmentation ONNX model")
    
    # Hyperparameters
    parser.add_argument("--conf_thres", type=float, default=50.0, help="Confidence percentile threshold (0-100)")
    parser.add_argument("--predict_motion", action="store_true", help="Enable motion mask prediction")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    try:
        engine = MoReInference(args)
        engine.run()
    except Exception as e:
        print(f"[Error] {e}")
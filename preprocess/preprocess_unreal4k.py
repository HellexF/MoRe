import argparse
import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from PIL import Image


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unreal4k_dir",
        default="/lpai/dataset/action-model-group-data-weight/0-1-0/datasets/unrealstereo4k/data4k",
    )
    parser.add_argument(
        "--output_dir",
        default="/lpai/volumes/base-3da-ali-sh-mix/zhangwq/dataset/vggt/unrealstereo4k",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of threads")
    return parser


def parse_extrinsics(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        intrinsics_data = list(map(float, lines[0].strip().split()))
        intrinsics_matrix = np.array(intrinsics_data).reshape(3, 3)
        pose_data = list(map(float, lines[1].strip().split()))
        pose_matrix = np.array(pose_data).reshape(3, 4)
        cam2world = np.eye(4)
        cam2world[:3] = pose_matrix
        cam2world = np.linalg.inv(cam2world)
        return intrinsics_matrix, cam2world


def process_frame(i, subscene, frame_dir, outdir, env):
    rgb_dir = osp.join(frame_dir, f"Image{subscene}")
    disp_dir = osp.join(frame_dir, f"Disp{subscene}")
    rgb_path = osp.join(rgb_dir, f"{i:05d}.png")
    disp_path = osp.join(disp_dir, f"{i:05d}.npy")
    ext_path0 = osp.join(frame_dir, "Extrinsics0", f"{i:05d}.txt")
    ext_path1 = osp.join(frame_dir, "Extrinsics1", f"{i:05d}.txt")

    out_env_dir = osp.join(outdir, env, subscene)
    os.makedirs(out_env_dir, exist_ok=True)
    out_rgb_path = osp.join(out_env_dir, f"{i:05d}_rgb.png")
    out_depth_path = osp.join(out_env_dir, f"{i:05d}_depth.npy")
    out_cam_path = osp.join(out_env_dir, f"{i:05d}.npz")

    try:
        K0, c2w0 = parse_extrinsics(ext_path0)
        K1, c2w1 = parse_extrinsics(ext_path1)
        if subscene == "0":
            K, c2w = K0, c2w0
        else:
            K, c2w = K1, c2w1

        img = Image.open(rgb_path).convert("RGB")
        disp = np.load(disp_path).astype(np.float32)
        baseline = (np.linalg.inv(c2w0) @ c2w1)[0, 3]
        depth = baseline * K[0, 0] / disp

        img.save(out_rgb_path)
        np.save(out_depth_path, depth)
        np.savez(out_cam_path, intrinsics=K, cam2world=np.linalg.inv(c2w))
    except Exception as e:
        print(f"[Warning] Failed to process {env}/{subscene}/{i:05d}: {e}")


def main(rootdir, outdir, num_workers):
    os.makedirs(outdir, exist_ok=True)
    envs = [f for f in sorted(os.listdir(rootdir)) if os.path.isdir(osp.join(rootdir, f))]

    for env in tqdm(envs, desc="Environments"):
        if env == ".huggingface":
            continue
        frame_dir = osp.join(rootdir, env)

        for subscene in ["0", "1"]:
            rgb_dir = osp.join(frame_dir, f"Image{subscene}")
            frame_num = len(os.listdir(rgb_dir))

            tasks = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for i in range(frame_num):
                    tasks.append(
                        executor.submit(process_frame, i, subscene, frame_dir, outdir, env)
                    )

                for _ in tqdm(as_completed(tasks), total=frame_num, desc=f"{env}-{subscene}"):
                    pass


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.unreal4k_dir, args.output_dir, args.num_workers)
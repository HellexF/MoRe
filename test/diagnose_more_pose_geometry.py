#!/usr/bin/env python3
"""Diagnose MoRe depth/camera/world-point consistency using saved predictions.

This is a read-only diagnostic for MoRe sources and predictions. It does not
register, move, rewrite, or rescale any PLY or pose file.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}


def frame_data(root: Path, idx: int):
    E = np.asarray(np.load(root / 'extrinsic' / f'{idx:05d}.npy'), dtype=np.float64)
    K = np.asarray(np.load(root / 'intrinsic' / f'{idx:05d}.npy'), dtype=np.float64)
    D = np.asarray(np.load(root / 'depth' / f'{idx:05d}.npy'), dtype=np.float32).squeeze()
    if E.shape != (3, 4) or K.shape != (3, 3) or D.ndim != 2:
        raise ValueError(f'frame {idx}: unexpected shapes: E={E.shape}, K={K.shape}, D={D.shape}')
    if not (np.isfinite(E).all() and np.isfinite(K).all()):
        raise ValueError(f'frame {idx}: nonfinite camera matrices')
    return E, K, D


def camera_center(E):
    R, t = E[:, :3], E[:, 3]
    return -R.T @ t


def angle_between(R0, R1):
    cosang = np.clip((np.trace(R1 @ R0.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosang)))


def unproject(uv, D, K):
    h, w = D.shape
    if len(uv) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    x = uv[:, 0].astype(np.float32).reshape(1, -1)
    y = uv[:, 1].astype(np.float32).reshape(1, -1)
    d = cv2.remap(D, x, y, interpolation=cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT, borderValue=0).reshape(-1).astype(np.float64)
    return np.stack(((uv[:, 0] - K[0, 2]) * d / K[0, 0],
                     (uv[:, 1] - K[1, 2]) * d / K[1, 1], d), axis=-1)


def world_from_w2c(Xcam, E):
    # Row-vector form of X_world = R.T @ (X_camera - t)
    return (Xcam - E[:3, 3]) @ E[:3, :3]


def world_if_c2w(Xcam, E):
    # A *diagnostic alternate*, NOT a recommended fix.
    return Xcam @ E[:3, :3].T + E[:3, 3]


def pct_stats(err, scale):
    if len(err) == 0:
        return None
    return {
        'median': float(np.median(err)),
        'p90': float(np.percentile(err, 90)),
        'median_pct_scene_depth': float(100.0 * np.median(err) / max(scale, 1e-10)),
    }


def get_processed_images(repo, imagepaths):
    sys.path.insert(0, str(repo))
    from vggt.utils.load_fn import load_and_preprocess_images
    images_t = load_and_preprocess_images([str(p) for p in imagepaths])
    return (images_t.mul(255).clamp(0, 255).to(dtype=images_t.dtype)
            .byte().permute(0, 2, 3, 1).cpu().numpy())


def matched_keypoints(img0, img1, sift):
    gray0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    kp0, desc0 = sift.detectAndCompute(gray0, None)
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    if desc0 is None or desc1 is None or len(desc0) < 2 or len(desc1) < 2:
        return np.empty((0, 2)), np.empty((0, 2)), 0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(desc0, desc1, k=2)
    good = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return np.empty((0, 2)), np.empty((0, 2)), len(good)
    uv0 = np.array([kp0[m.queryIdx].pt for m in good], dtype=np.float64)
    uv1 = np.array([kp1[m.trainIdx].pt for m in good], dtype=np.float64)
    _, inlier_mask = cv2.findFundamentalMat(uv0, uv1, cv2.FM_RANSAC, 1.5, 0.995)
    if inlier_mask is None:
        return np.empty((0, 2)), np.empty((0, 2)), len(good)
    keep = inlier_mask.reshape(-1).astype(bool)
    return uv0[keep], uv1[keep], len(good)


def filter_static(uv0, uv1, root, i, j):
    keep = np.ones(len(uv0), dtype=bool)
    for idx, uv in ((i, uv0), (j, uv1)):
        u = np.round(uv[:, 0]).astype(int)
        v = np.round(uv[:, 1]).astype(int)
        sky_file = root / 'glb' / 'sky_masks' / f'{idx}.png'
        motion_file = root / 'masks' / f'mask_{idx:05d}.png'
        for file, limit, is_greater in ((sky_file, 127, True), (motion_file, 127, False)):
            if not file.exists():
                continue
            arr = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            if arr is None:
                continue
            if is_greater:
                arr_values = arr[np.clip(v, 0, arr.shape[0]-1), np.clip(u, 0, arr.shape[1]-1)]
                keep &= (arr_values > limit)
            else:
                arr_values = arr[np.clip(v, 0, arr.shape[0]-1), np.clip(u, 0, arr.shape[1]-1)]
                keep &= (arr_values < limit)
    return uv0[keep], uv1[keep]


def analyze_pair(root, imgs, i, j, sift, use_masks):
    E0, K0, D0 = frame_data(root, i)
    E1, K1, D1 = frame_data(root, j)
    uv0, uv1, initial_matches = matched_keypoints(imgs[i], imgs[j], sift)
    if use_masks and len(uv0):
        uv0, uv1 = filter_static(uv0, uv1, root, i, j)
    n_feature_inliers = len(uv0)
    if len(uv0) < 8:
        return {'frames': [i, j], 'ratio_test_matches': initial_matches,
                'fundamental_inliers_after_mask': n_feature_inliers,
                'note': 'insufficient matched features; try adjacent frames or no mask'}
    X0, X1 = unproject(uv0, D0, K0), unproject(uv1, D1, K1)
    valid = np.isfinite(X0).all(axis=1) & np.isfinite(X1).all(axis=1) & (X0[:, 2] > 1e-6) & (X1[:, 2] > 1e-6)
    X0, X1 = X0[valid], X1[valid]
    if len(X0) < 8:
        return {'frames': [i, j], 'ratio_test_matches': initial_matches,
                'fundamental_inliers_after_mask': n_feature_inliers,
                'note': 'insufficient positive-depth correspondences'}
    scale = float(np.median(np.concatenate([X0[:, 2], X1[:, 2]])))
    results = {
        'frames': [i, j], 'ratio_test_matches': initial_matches,
        'fundamental_inliers_after_mask': n_feature_inliers,
        'valid_3d_matches': len(X0), 'median_scene_depth': scale,
        'camera_local': pct_stats(np.linalg.norm(X0-X1, axis=1), scale),
        'expected_w2c_inverse': pct_stats(np.linalg.norm(world_from_w2c(X0,E0)-world_from_w2c(X1,E1), axis=1),scale),
        'alternative_c2w_interpretation': pct_stats(np.linalg.norm(world_if_c2w(X0,E0)-world_if_c2w(X1,E1), axis=1),scale),
        'camera_baseline': float(np.linalg.norm(camera_center(E1)-camera_center(E0))),
        'rotation_deg': angle_between(E0[:,:3], E1[:,:3]),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path, help='e.g. /workspace/MoRe/results/input003_full_glbfix')
    parser.add_argument('--images', required=True, type=Path, help='original input images directory')
    parser.add_argument('--repo', type=Path, default=Path('/workspace/MoRe/code/MoRe'))
    parser.add_argument('--pairs', nargs='*', help='frame index pairs like 0:1 0:10 10:20')
    parser.add_argument('--use-masks', action='store_true', help='filter sky and motion positive pixels; may remove too many')
    parser.add_argument('--output', type=Path, help='optional JSON report path')
    args = parser.parse_args()
    root = args.results
    frames = sorted(p for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    n = len(frames)
    if n < 2:
        parser.error('need at least two images')
    Es = []
    depth_medians = []
    for i in range(n):
        E, K, D = frame_data(root, i)
        Es.append(E)
        valid = D[np.isfinite(D) & (D>0)]
        depth_medians.append(float(np.median(valid)) if valid.size else float('nan'))
    centers = np.array([camera_center(E) for E in Es])
    rotations = np.array([angle_between(Es[0][:,:3], E[:,:3]) for E in Es])
    orth_err = np.array([np.linalg.norm(E[:,:3].T @ E[:,:3] - np.eye(3)) for E in Es])
    base = np.linalg.norm(centers-centers[0], axis=1)
    report = {
        'frames': n,
        'first_center': centers[0].tolist(), 'last_center': centers[-1].tolist(),
        'camera_center_max_displacement_from_first': float(base.max()),
        'camera_center_final_displacement_from_first': float(base[-1]),
        'camera_path_length': float(np.sum(np.linalg.norm(np.diff(centers,axis=0),axis=1))),
        'max_rotation_deg_from_first': float(rotations.max()),
        'final_rotation_deg_from_first': float(rotations[-1]),
        'max_extrinsic_rotation_orthogonality_error': float(orth_err.max()),
        'median_predicted_depth': float(np.nanmedian(depth_medians)),
        'pairs': [],
    }
    print('\n=== CAMERA POSE SUMMARY ===')
    for key, value in report.items():
        if key != 'pairs':
            print(f'{key}: {value}')
    if args.pairs:
        pairs = []
        for s in args.pairs:
            if ':' not in s:
                parser.error(f'invalid pair: {s}')
            i, j = map(int, s.split(':',1))
            if i<0 or j<0 or i>=n or j>=n or i==j:
                parser.error(f'invalid pair {s}: available indices 0..{n-1}')
            pairs.append((i,j))
    else:
        candidates = [(0,1),(0,5),(0,10),(10,20),(n//2, min(n-1,n//2+5)),(max(0,n-6),n-1)]
        pairs = list(dict.fromkeys((i,j) for i,j in candidates if i<j<n))
    imgs = get_processed_images(args.repo, frames)
    print(f'preprocessed images: {imgs.shape}')
    sift = cv2.SIFT_create(nfeatures=4000)
    print('\n=== MATCHED STATIC-SCENE 3D CONSISTENCY (not an accuracy ground truth) ===')
    for i, j in pairs:
        result = analyze_pair(root, imgs, i, j, sift, args.use_masks)
        report['pairs'].append(result)
        print(f'frame {i:03d}->{j:03d}: matches={result.get("valid_3d_matches",0)}')
        if 'note' in result:
            print('   ', result['note'])
        else:
            print(f'    baseline={result["camera_baseline"]:.4f} rotation={result["rotation_deg"]:.2f}°')
            for key in ('camera_local','expected_w2c_inverse','alternative_c2w_interpretation'):
                s = result[key]
                print(f'    {key:29s} med={s["median"]:.4f}  p90={s["p90"]:.4f}  med/depth={s["median_pct_scene_depth"]:.2f}%')
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        print(f'\nSaved: {args.output}')
    print('\nNotes: matching/occlusion/dynamic objects can affect errors. No PLY files were modified.')

if __name__ == '__main__':
    main()

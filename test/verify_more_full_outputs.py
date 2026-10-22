#!/usr/bin/env python3
"""Audit a MoRe Full-Attention run. Does not change any results or Git files.
Usage: python verify_more_full_outputs.py <input_images_dir> <output_dir>
"""
import sys
from pathlib import Path
import numpy as np

if len(sys.argv) != 3:
    raise SystemExit(f'用法: python {Path(sys.argv[0]).name} <input_images_dir> <output_dir>')
images_dir, root = map(Path, sys.argv[1:])
inputs = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
if not inputs:
    raise SystemExit(f'无输入帧: {images_dir}')

issues = []
empty_ply_frames = []
for i, _ in enumerate(inputs):
    name = f'{i:05d}'
    required = {
        'PLY': root / f'frame_{name}.ply',
        'depth': root / 'depth' / f'{name}.npy',
        'extrinsic': root / 'extrinsic' / f'{name}.npy',
        'intrinsic': root / 'intrinsic' / f'{name}.npy',
        'mask': root / 'masks' / f'mask_{name}.png',
    }
    for label, f in required.items():
        if not f.is_file() or f.stat().st_size == 0:
            issues.append(f'帧 {i}: {label} 文件缺失或为空: {f}')

    if required['PLY'].is_file():
        # Read only the ASCII PLY header; large point clouds stay out of memory.
        try:
            with required['PLY'].open('rb') as fp:
                first = fp.readline().strip()
                if first != b'ply':
                    raise ValueError('invalid PLY magic')
                count = None
                for j in range(100):
                    line = fp.readline().decode('ascii').strip()
                    if line.startswith('element vertex '):
                        count = int(line.split()[2])
                    if line == 'end_header':
                        break
                else:
                    raise ValueError('PLY header missing end_header')
                if count is None:
                    raise ValueError('PLY header missing vertex count')
                if count == 0:
                    empty_ply_frames.append(i)
        except Exception as e:
            issues.append(f'帧 {i}: PLY 头检查失败: {e}')

    for label in ('depth', 'extrinsic', 'intrinsic'):
        f = required[label]
        if f.is_file():
            try:
                arr = np.load(f, allow_pickle=False)
                if arr.size == 0 or not np.isfinite(arr).all():
                    issues.append(f'帧 {i}: {label} 为空或含 NaN/Inf: {f}')
                if label == 'depth' and arr.ndim != 2:
                    issues.append(f'帧 {i}: depth 维数不是 2: {arr.shape}')
                if label == 'extrinsic' and arr.shape != (3, 4):
                    issues.append(f'帧 {i}: extrinsic 形状异常: {arr.shape}')
                if label == 'intrinsic' and arr.shape != (3, 3):
                    issues.append(f'帧 {i}: intrinsic 形状异常: {arr.shape}')
            except Exception as e:
                issues.append(f'帧 {i}: {label} 数组读取失败: {e}')

print(f'输入帧数: {len(inputs)}')
print(f'预期输出: 每类 {len(inputs)} 个 PLY / depth / extrinsic / intrinsic / mask')
if empty_ply_frames:
    print(f'警告：这些帧的 PLY 顶点数为零: {empty_ply_frames}')
if issues:
    print(f'检查未通过（{len(issues)} 个问题）：')
    for issue in issues:
        print(' -', issue)
    raise SystemExit(1)
if len(empty_ply_frames) == len(inputs):
    raise SystemExit('文件存在但所有 PLY 均无有效顶点，不能判定结果有效')
print('导出结构及数值基本检查通过；仍需进行视觉/几何质量评估。')

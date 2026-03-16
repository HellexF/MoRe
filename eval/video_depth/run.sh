#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
set -e

workdir='../../'
model_name='more'
model_weights="pretrained/full/model.pt"
datasets=('sintel' 'bonn' 'kitti')

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/video_depth/${data}_${model_name}"
    echo "$output_dir"
    accelerate launch --main_process_port 29600 --num_processes 2  eval/video_depth/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 518
    python eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --max_depth 120 \
    --align "scale"
done

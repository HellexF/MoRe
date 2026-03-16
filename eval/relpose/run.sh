#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
set -e

workdir='../../'
model_name='more'
model_weights="pretrained/full/model.pt"
datasets=('sintel' 'bonn' 'tum' 'scannet')


for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/relpose/${data}_${model_name}"
    echo "$output_dir"
    accelerate launch --num_processes 5 --main_process_port 29558 eval/relpose/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --config_name "full_attn_test_v11" \
        --size 512
done



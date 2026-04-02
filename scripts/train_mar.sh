#!/bin/bash
# Multi-GPU training script for xAR Pixel models
# Usage: bash scripts/train_pixel_multi_gpu.sh

# Wandb configuration
wandb_key="4ab8d4a0db9aec6c80956ccf58616de15392a463"
wandb_project="ssl2gen"
wandb_entity="qkrwlgh0314"

# Dataset options: cifar10-hf, tiny-imagenet-hf, mnist-hf, imagenet
run_name="0401_marssllatent-base_vae_tinyimgnet_repa"
exps_dir="./ho_mar_0311"
config=$exps_dir/$run_name/config.yaml

# ongoing: high-resolution good? 
torchrun --nnodes=1 --nproc_per_node=8 --master_port=11121 main_mar.py \
    --config "$config" \
    --dtype "bf16" \
    --num_workers 8 \
    --eval_freq 1 \
    --eval_bsz 49 \
    --save_last_freq 4 \
    --num_iter 64 \
    --cfg 2.5 \
    --output_dir $exps_dir \
    --run_name "${run_name}" \
    --online_eval \
    --resume_last \
    # --wandb_key "${wandb_key}" \
    # --wandb_project "${wandb_project}" \
    # --wandb_entity "${wandb_entity}" \

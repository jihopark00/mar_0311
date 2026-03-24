#!/bin/bash
# Multi-GPU training script for xAR Pixel models
# Usage: bash scripts/train_pixel_multi_gpu.sh

# Wandb configuration
wandb_key="4ab8d4a0db9aec6c80956ccf58616de15392a463"
wandb_project="ssl2gen"
wandb_entity="qkrwlgh0314"

# Dataset options: cifar10-hf, tiny-imagenet-hf, mnist-hf, imagenet
dataset="tiny-imagenet-hf"
# dataset="mnist-hf"
run_name="marflowfull_base_224_debug_jit"

# ongoing: high-resolution good? 
# TODO: clusters=8
# Multi-GPU training (2 GPUs)
torchrun --nnodes=1 --nproc_per_node=4 --master_port=22212 main_mar_pixel.py \
    --config "configs/jit_base_224.yaml" \
    --dtype "bf16" \
    --batch_size 90 \
    --epochs 10000 \
    --lr 1e-4 \
    --warmup_epochs 0 \
    --num_workers 8 \
    --eval_freq 1 \
    --eval_bsz 36 \
    --num_iter 256 \
    --cfg 2.5 \
    --save_last_freq 4 \
    --dataset "${dataset}" \
    --output_dir "ho_mar_0311" \
    --run_name "${run_name}" \
    --online_eval \
    --resume_last \
    --debug_one_image
    # --wandb_key "${wandb_key}" \
    # --wandb_project "${wandb_project}" \
    # --wandb_entity "${wandb_entity}" \

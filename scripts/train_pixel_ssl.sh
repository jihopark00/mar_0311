#!/bin/bash
# Multi-GPU training script for xAR Pixel models
# Usage: bash scripts/train_pixel_multi_gpu.sh

# Wandb configuration
wandb_key="4ab8d4a0db9aec6c80956ccf58616de15392a463"
wandb_project="ssl2gen"
wandb_entity="qkrwlgh0314"

# Dataset options: cifar10-hf, tiny-imagenet-hf, mnist-hf, imagenet
# dataset="tiny-imagenet-hf"
dataset="mnist-hf"
run_name="marssl_pixel_256_mnist"

# ongoing: high-resolution good? 
# TODO: clusters=8
# Multi-GPU training (2 GPUs)
torchrun --nnodes=1 --nproc_per_node=8 --master_port=12322 main_mar_pixel.py \
    --config "configs/marflow_ssl_256.yaml" \
    --dtype "bf16" \
    --batch_size 28 \
    --epochs 10000 \
    --lr 1e-4 \
    --warmup_epochs 0 \
    --num_workers 8 \
    --eval_freq 4 \
    --eval_bsz 16 \
    --num_iter 128 \
    --cfg 2.5 \
    --save_last_freq 1 \
    --dataset "${dataset}" \
    --output_dir "ho_mar_0311" \
    --run_name "${run_name}" \
    --wandb_key "${wandb_key}" \
    --wandb_project "${wandb_project}" \
    --wandb_entity "${wandb_entity}" \
    --online_eval \
    --resume_last \
    # --debug_one_image

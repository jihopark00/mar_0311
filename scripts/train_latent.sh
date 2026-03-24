#!/bin/bash
# Multi-GPU training script for xAR Pixel models
# Usage: bash scripts/train_pixel_multi_gpu.sh

# Wandb configuration
wandb_key="4ab8d4a0db9aec6c80956ccf58616de15392a463"
wandb_project="ssl2gen"
wandb_entity="qkrwlgh0314"

# Dataset options: cifar10-hf, tiny-imagenet-hf, mnist-hf, imagenet
run_name="0323_marssllatent_256_cifar10"
exps_dir="./ho_mar_0311"
config=$exps_dir/$run_name/config.yaml

# ongoing: high-resolution good? 
# TODO: clusters=8
# Multi-GPU training (2 GPUs)
torchrun --nnodes=1 --nproc_per_node=4 --master_port=33221 main_mar_latent.py \
    --vae_ckpt /home/ljeadec31/opt/ssl2gen-top/mar_0311/pretrained_models/vae/kl16.ckpt \
    --config "$config" \
    --dtype "bf16" \
    --num_workers 8 \
    --eval_freq 10 \
    --eval_bsz 36 \
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

#!/bin/bash
# Multi-GPU evaluation script for MAR models
# Usage: bash scripts/test.sh

exps_dir="./ho_mar_0311"
run_names=( 0409_mar-base_dec-only ) # 0409_mar-base_dec-only) # "0405_marssllatent-base_imgnet") #  "0403_mar-base_imgnet-cache"
train_steps=( 40)

for run_name in "${run_names[@]}"; do
    for train_step in "${train_steps[@]}"; do
        echo "========== Evaluating run_name=${run_name} train_step=${train_step} =========="
        torchrun --nnodes=1 --nproc_per_node=4 --master_port=33332 eval.py \
            --exps_dir "$exps_dir" \
            --run_name "$run_name" \
            --train_step "$train_step" \
            --num_images 10000 \
            --batch_size 64 \
            --num_iter 64 \
            --cfg 1.0 \
            --cfg_schedule linear \
            --temperature 1.0 \
            --dtype fp32 \
            --seed 0 \
            --save_samples_dir "${exps_dir}/${run_name}/eval_samples_${train_step}" \
            --csv_file "${exps_dir}/eval_results.csv" \
            --fid_stats "fid_stats/adm_in256_stats.npz" \
            --clean_samples \
            # --no_ema \
            # --save_npz
    done
done

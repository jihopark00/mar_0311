#!/bin/bash


IMAGENET_PATH='/dataset/imagenet'
CACHED_PATH='/scratch2/ljeadec31/imagenet_dinov2_vitg14_reg'

python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 --master_port=11113  \
    main_cache_dinov2.py \
    --dinov2_repo facebookresearch/dinov2 \
    --dinov2_name dinov2_vitg14_reg \
    --dinov2_input_size 224 \
    --save_dtype fp16 \
    --batch_size 512 \
    --data_path ${IMAGENET_PATH} --cached_path ${CACHED_PATH}

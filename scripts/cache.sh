#!/bin/bash


# /scratch2/ljeadec31/imagenet_mar_sdvae



IMAGENET_PATH='/dataset/imagenet'
CACHED_PATH='/scratch2/ljeadec31/imagenet_mar_sdvae'

python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 --master_port=11112  \
    main_cache.py \
    --img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 \
    --batch_size 256 \
    --data_path ${IMAGENET_PATH} --cached_path ${CACHED_PATH}
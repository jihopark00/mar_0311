
# cd ..
python visualize_cache.py \
    --config configs/config.yaml \
    --cached_path /scratch2/ljeadec31/imagenet_mar_sdvae \
    --feat_cached_root /scratch2/ljeadec31/imagenet_dinov2_vitg14_reg \
    --data_path /dataset/imagenet/train \
    --n_samples 64 \
    --out vis_cache.png

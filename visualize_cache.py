"""
Visualize cached dataset: decoded VAE latent | img_pixel | feat PCA.

Checks that hflip is applied consistently across (cached_latent, img_pixel, feat).

Usage:
    cd /iopsstor/scratch/cscs/junwan/ho/ssl2gen/mar_0311
    python visualize_cache.py \
        --config ho/0413_mardino-giant_bf16/config.yaml \
        --cached_path /iopsstor/scratch/cscs/junwan/ho/imagenet-sdvae \
        --feat_cached_root /iopsstor/scratch/cscs/junwan/ho/imagenet_dinov2_vitg14_reg \
        --data_path /iopsstor/scratch/cscs/junwan/imagenet/imagenet-train \
        --n_samples 16 \
        --out vis_cache.png
"""

import argparse
import os
import sys
import numpy as np
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ── repo path ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from util.loader import CachedLatentDataset


class CachedLatentDatasetWithFlip(CachedLatentDataset):
    """Thin wrapper that appends a flip flag to each sample."""

    def __getitem__(self, index):
        items = super().__getitem__(index)
        # Detect flip: reload the original image without flip and compare
        path, _ = self.samples[index]
        rel = os.path.relpath(path, self.root)
        img_rel = rel[: -len('.npz')]
        orig = self.Image.open(
            os.path.join(self.data_path, img_rel)
        ).convert('RGB')
        if self.img_transform is not None:
            orig = self.img_transform(orig)
        # items[2] is img_pix (possibly flipped)
        flipped = not torch.equal(items[2], orig)
        return (*items, flipped)


# ── helpers ────────────────────────────────────────────────────────────────────

def feat_pca_rgb(feats: torch.Tensor, n_reg: int = 4) -> torch.Tensor:
    """
    feats : [B, seq_len, D]  (cls + registers + spatial patches)
    Returns [B, H, W, 3] float in [0, 1], where H*W = n_spatial_patches.
    PCA is fitted independently per image.
    """
    B, seq_len, D = feats.shape
    n_skip = 1 + n_reg          # cls + register tokens
    spatial = feats[:, n_skip:, :]          # [B, N, D]
    N = spatial.shape[1]
    H = W = int(N ** 0.5)
    assert H * W == N, f"spatial token count {N} is not a perfect square"

    result = np.empty((B, H, W, 3), dtype=np.float64)
    for b in range(B):
        flat = spatial[b].float().numpy()               # [N, D]
        pcs = PCA(n_components=3).fit_transform(flat)   # [N, 3]
        for c in range(3):
            lo, hi = pcs[:, c].min(), pcs[:, c].max()
            pcs[:, c] = (pcs[:, c] - lo) / (hi - lo + 1e-8)
        result[b] = pcs.reshape(H, W, 3)

    return result                                       # [B, H, W, 3]


def to_uint8(t: torch.Tensor) -> np.ndarray:
    """[3, H, W] float [0,1] → [H, W, 3] uint8"""
    return (t.permute(1, 2, 0).cpu().float().clamp(0, 1).numpy() * 255).astype(np.uint8)


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',          required=True)
    p.add_argument('--cached_path',     required=True)
    p.add_argument('--feat_cached_root',default=None)
    p.add_argument('--data_path',       required=True)
    p.add_argument('--n_samples',       type=int, default=16,
                   help='number of samples to visualise (≤ batch_size)')
    p.add_argument('--batch_size',      type=int, default=32)
    p.add_argument('--seed',            type=int, default=0)
    p.add_argument('--out',             default='vis_cache.png')
    p.add_argument('--device',          default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    vae_name   = cfg['vae']
    vae_config = cfg['vae_config']
    latent_mean = vae_config.get('latent_mean', 0.0)
    latent_std  = vae_config.get('latent_std',  1.0)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ── VAE ────────────────────────────────────────────────────────────────────
    import vae as vae_module
    vae_cls = getattr(vae_module, vae_name)
    vae = vae_cls(**vae_config).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"Loaded VAE: {vae_name}")

    # ── Dataset ────────────────────────────────────────────────────────────────
    img_size = cfg['model_config'].get('img_size', 256)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    dataset = CachedLatentDatasetWithFlip(
        root=args.cached_path,
        data_path=args.data_path,
        transform=transform,
        latent_mean=latent_mean,
        latent_std=latent_std,
        feat_cached_root=args.feat_cached_root,
    )
    print(f"Dataset size: {len(dataset)}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # ── Get one batch ──────────────────────────────────────────────────────────
    batch = next(iter(loader))
    # Last element is always the flip flag from our wrapper
    has_feat = (len(batch) == 5)

    z       = batch[0][:args.n_samples]   # [N, C, H, W]
    labels  = batch[1][:args.n_samples]
    img_pix = batch[2][:args.n_samples]   # [N, 3, img_size, img_size]
    if has_feat:
        feat  = batch[3][:args.n_samples]    # [N, seq_len, D]
        flips = batch[4][:args.n_samples]
    else:
        feat  = None
        flips = batch[3][:args.n_samples]

    print(f"z shape      : {z.shape}")
    print(f"img_pix shape: {img_pix.shape}")
    if feat is not None:
        print(f"feat shape   : {feat.shape}")

    n = z.shape[0]

    # ── Decode latent via VAE ──────────────────────────────────────────────────
    # vae.decode() handles denormalization internally (do_normalization=True)
    with torch.no_grad():
        decoded = vae.decode(z.to(device)).cpu().clamp(0, 1)  # [N, 3, H, W]

    # ── PCA for cached feat ───────────────────────────────────────────────────
    if feat is not None:
        feat_rgb = feat_pca_rgb(feat)   # [N, H_p, W_p, 3]

    # ── Live DINOv2 feat from pixel ───────────────────────────────────────────
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
    dino = dino.to(device).eval()
    for p in dino.parameters():
        p.requires_grad_(False)
    print("Loaded DINOv2 ViT-G/14 reg (live)")

    dino_size = 224
    dino_norm = transforms.Compose([
        transforms.Resize(dino_size),
        transforms.CenterCrop(dino_size),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_dino = dino_norm(img_pix)  # [N, 3, dino_size, dino_size]
    # denormalize for visualization
    dino_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    dino_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_dino_vis = (img_dino * dino_std + dino_mean).clamp(0, 1)
    # import pdb ; pdb.set_trace()
    with torch.no_grad():
        dino_out = dino.forward_features(img_dino.to(device))
        live_feat = dino_out['x_norm_patchtokens'].cpu()   # [N, H*W, D]
    # Prepend dummy cls+reg tokens so feat_pca_rgb can skip them uniformly
    n_reg = 4
    dummy_prefix = torch.zeros(live_feat.shape[0], 1 + n_reg, live_feat.shape[2])
    live_feat_full = torch.cat([dummy_prefix, live_feat], dim=1)
    live_feat_rgb = feat_pca_rgb(live_feat_full, n_reg=n_reg)  # [N, H_p, W_p, 3]
    print(f"live_feat shape: {live_feat.shape}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    col_titles = ['decoded latent (VAE)', f'img_pixel ({img_size})', f'img_dino_input ({dino_size})']
    if has_feat:
        col_titles.append('feat PCA (cached)')
    col_titles.append('feat PCA (live DINOv2-G)')
    n_cols = len(col_titles)

    fig, axes = plt.subplots(n, n_cols, figsize=(n_cols * 3, n * 3))
    if n == 1:
        axes = axes[None]   # ensure 2-D indexing

    for i in range(n):
        # decoded latent
        axes[i, 0].imshow(to_uint8(decoded[i]))
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title(col_titles[0], fontsize=9)

        # img_pixel
        axes[i, 1].imshow(to_uint8(img_pix[i]))
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title(col_titles[1], fontsize=9)

        # img_dino_input (denormalized)
        axes[i, 2].imshow(to_uint8(img_dino_vis[i]))
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 2].set_title(col_titles[2], fontsize=9)

        # cached feat PCA
        col = 3
        if has_feat:
            axes[i, col].imshow(feat_rgb[i])
            axes[i, col].axis('off')
            if i == 0:
                axes[i, col].set_title(col_titles[col], fontsize=9)
            col += 1

        # live DINOv2 feat PCA
        axes[i, col].imshow(live_feat_rgb[i])
        axes[i, col].axis('off')
        if i == 0:
            axes[i, col].set_title(col_titles[col], fontsize=9)

        # row label: class index + flip flag
        flip_str = 'flip' if flips[i] else 'orig'
        axes[i, 0].text(-0.02, 0.5, f'cls={labels[i].item()}\n[{flip_str}]',
                         fontsize=7, ha='right', va='center',
                         transform=axes[i, 0].transAxes)

    plt.suptitle('Cache visualisation — check hflip consistency', fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved → {args.out}")


if __name__ == '__main__':
    main()

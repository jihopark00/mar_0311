"""
JiT (Just image Transformer) model for MAR-style training.

This module wraps the JiT model with flow matching training/sampling,
providing the same interface as MAR for compatibility with main_mar_pixel.py.

Usage:
    model = jit_base(img_size=256, class_num=1000)
    loss = model(imgs, labels)  # training
    samples = model.sample_tokens(bsz=16, labels=labels)  # generation
"""

from functools import partial

import math
import torch
import torch.nn as nn
from tqdm import tqdm

from models.model_jit import JiT_models


class JiTMAR(nn.Module):
    """
    JiT model wrapped with flow matching for MAR-style training.

    Provides the same interface as MAR:
    - forward(imgs, labels) -> loss
    - sample_tokens(bsz, num_iter, cfg, cfg_schedule, labels, temperature) -> samples
    """

    def __init__(
        self,
        img_size=256,
        vae_stride=1,
        patch_size=16,
        class_num=1000,
        label_drop_prob=0.1,
        attn_dropout=0.1,
        proj_dropout=0.1,
        # Flow matching params
        P_mean=0.0,
        P_std=1.0,
        t_eps=0.02,
        noise_scale=1.0,
        # Sampling params
        num_sampling_steps=50,
        sampling_method="euler",
        time_shift_scale=1.0,
        # Model architecture
        model_type='JiT-L/16',
        grad_checkpointing=False,
        # Unused MAR params (for config compatibility)
        encoder_embed_dim=None,
        encoder_depth=None,
        encoder_num_heads=None,
        decoder_embed_dim=None,
        decoder_depth=None,
        decoder_num_heads=None,
        mlp_ratio=None,
        vae_embed_dim=None,
        mask_ratio_min=None,
        buffer_size=None,
        diffusion_batch_mul=None,
        diffloss_class=None,
        diffloss_kwargs=None,
    ):
        super().__init__()

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.num_classes = class_num
        self.label_drop_prob = label_drop_prob
        self.grad_checkpointing = grad_checkpointing

        # Flow matching params
        self.P_mean = P_mean
        self.P_std = P_std
        self.t_eps = t_eps
        self.noise_scale = noise_scale

        # Sampling params
        self.num_sampling_steps = num_sampling_steps
        self.sampling_method = sampling_method
        self.time_shift_scale = time_shift_scale

        # Compute latent size after VAE
        self.latent_size = img_size // vae_stride

        # Build JiT model
        # Select model based on model_type or build from patch_size
        if model_type in JiT_models:
            self.net = JiT_models[model_type](
                input_size=self.latent_size,
                in_channels=3,
                num_classes=class_num,
                attn_drop=attn_dropout,
                proj_drop=proj_dropout,
            )
        else:
            # Fallback: use patch_size to select model
            model_key = f'JiT-L/{patch_size}'
            if model_key in JiT_models:
                self.net = JiT_models[model_key](
                    input_size=self.latent_size,
                    in_channels=3,
                    num_classes=class_num,
                    attn_drop=attn_dropout,
                    proj_drop=proj_dropout,
                )
            else:
                raise ValueError(f"Unknown model type: {model_type} and no matching JiT model for patch_size={patch_size}")

        # For MAR interface compatibility - create a fake diffloss with net attribute
        self.diffloss = _FakeDiffLoss(self.net)

    def drop_labels(self, labels):
        """Drop labels for classifier-free guidance training."""
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        """Sample timesteps from logit-normal distribution."""
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, imgs, labels):
        """
        Training forward pass with flow matching loss.

        Args:
            imgs: [B, C, H, W] images (from VAE encoding)
            labels: [B] class labels

        Returns:
            Scalar loss value
        """
        # Drop labels for CFG training
        labels_dropped = self.drop_labels(labels) if self.training else labels

        # Sample timesteps
        t = self.sample_t(imgs.size(0), device=imgs.device).view(-1, *([1] * (imgs.ndim - 1)))

        # Sample noise
        e = torch.randn_like(imgs) * self.noise_scale

        # Create noisy sample: z = t * x + (1-t) * noise
        z = t * imgs + (1 - t) * e

        # Target velocity: v = (x - z) / (1 - t)
        v = (imgs - z) / (1 - t).clamp_min(self.t_eps)

        # Predict clean image
        x_pred = self.net(z, t.flatten(), labels_dropped)

        # Predicted velocity from x-prediction
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # L2 velocity loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss

    @torch.no_grad()
    def sample_tokens(
        self,
        bsz,
        num_iter=50, # not used
        cfg=1.0,
        cfg_schedule="linear",
        labels=None,
        temperature=1.0,
        progress=False
    ):
        """
        Generate samples using ODE integration.

        Args:
            bsz: Batch size
            num_iter: Number of integration steps (uses self.num_sampling_steps if not matching)
            cfg: Classifier-free guidance scale
            cfg_schedule: CFG schedule type (not used, for interface compatibility)
            labels: [B] class labels (None for unconditional)
            temperature: Noise temperature
            progress: Whether to show progress bar

        Returns:
            [B, C, H, W] generated images
        """
        device = next(self.parameters()).device

        # # Use num_iter as sampling steps
        # steps = num_iter if num_iter > 1 else self.num_sampling_steps
        steps = self.num_sampling_steps

        # Initialize with noise
        x = temperature * self.noise_scale * torch.randn(
            bsz, 3, self.latent_size, self.latent_size, device=device
        )

        # Setup labels
        if labels is None:
            labels = torch.full((bsz,), self.num_classes, device=device, dtype=torch.long)

        # CFG setup
        if cfg != 1.0:
            x = torch.cat([x, x], dim=0)
            labels_cond = labels
            labels_uncond = torch.full_like(labels, self.num_classes)
            labels = torch.cat([labels_cond, labels_uncond], dim=0)

        # Timesteps from 0 to 1 with time shift
        timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device)
        timesteps = timesteps / (self.time_shift_scale - (self.time_shift_scale - 1) * timesteps)

        # Select integration method
        if self.sampling_method == "euler":
            stepper = self._euler_step
        elif self.sampling_method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError(f"Unknown sampling method: {self.sampling_method}")

        # ODE integration
        indices = list(range(steps - 1))
        if progress:
            indices = tqdm(indices)

        for i in indices:
            t = timesteps[i]
            t_next = timesteps[i + 1]
            x = stepper(x, t, t_next, labels, cfg)

        # Last step with euler
        t = timesteps[-2]
        t_next = timesteps[-1]
        x = self._euler_step(x, t, t_next, labels, cfg)

        # Remove CFG duplicates
        if cfg != 1.0:
            x, _ = x.chunk(2, dim=0)

        return x

    @torch.no_grad()
    def _forward_sample(self, x, t, labels, cfg=1.0):
        """
        Forward pass for sampling with optional CFG.
        """
        bsz = x.shape[0]
        t_input = t.expand(bsz)
        t_view = t.view(1, 1, 1, 1).expand(bsz, 1, 1, 1)

        # x-prediction
        x_pred = self.net(x, t_input, labels)
        v_pred = (x_pred - x) / (1.0 - t_view).clamp_min(self.t_eps)

        if cfg != 1.0:
            # Split cond/uncond predictions
            v_cond, v_uncond = v_pred.chunk(2, dim=0)
            # CFG interpolation
            v_cfg = v_uncond + cfg * (v_cond - v_uncond)
            # Duplicate back for both halves
            v_pred = torch.cat([v_cfg, v_cfg], dim=0)

        return v_pred

    @torch.no_grad()
    def _euler_step(self, x, t, t_next, labels, cfg=1.0):
        """Euler integration step."""
        v_pred = self._forward_sample(x, t, labels, cfg)
        dt = t_next - t
        x_next = x + dt * v_pred
        return x_next

    @torch.no_grad()
    def _heun_step(self, x, t, t_next, labels, cfg=1.0):
        """Heun integration step (2nd order)."""
        dt = t_next - t

        # First euler step
        v_pred_t = self._forward_sample(x, t, labels, cfg)
        x_next_euler = x + dt * v_pred_t

        # Evaluate at next point
        v_pred_t_next = self._forward_sample(x_next_euler, t_next, labels, cfg)

        # Average velocities
        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        x_next = x + dt * v_pred
        return x_next


class _FakeDiffLoss(nn.Module):
    """Fake diffloss class to maintain interface compatibility with MAR."""
    def __init__(self, net):
        super().__init__()
        self.net = net


# Factory functions matching MAR style
def jit_base(**kwargs):
    """JiT-B/16 model."""
    model = JiTMAR(
        model_type='JiT-B/16',
        **kwargs
    )
    return model


def jit_large(**kwargs):
    """JiT-L/16 model."""
    model = JiTMAR(
        model_type='JiT-L/16',
        **kwargs
    )
    return model


def jit_huge(**kwargs):
    """JiT-H/16 model."""
    model = JiTMAR(
        model_type='JiT-H/16',
        **kwargs
    )
    return model


def jit_base_32(**kwargs):
    """JiT-B/32 model."""
    model = JiTMAR(
        model_type='JiT-B/32',
        **kwargs
    )
    return model


def jit_large_32(**kwargs):
    """JiT-L/32 model."""
    model = JiTMAR(
        model_type='JiT-L/32',
        **kwargs
    )
    return model


def jit_huge_32(**kwargs):
    """JiT-H/32 model."""
    model = JiTMAR(
        model_type='JiT-H/32',
        **kwargs
    )
    return model

"""
Training engine for xAR Latent models.

This module provides training and evaluation functions for xAR models
operating in latent space with a pretrained KL-VAE.
"""

import math
import sys
from typing import Iterable

import torch
import torchvision.utils as vutils

import util.misc as misc
import util.lr_sched as lr_sched
import copy
import numpy as np
import os


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def train_one_epoch(
        model,
        vae,
        model_params,
        ema_params,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        loss_scaler,
        log_writer=None,
        args=None,
        wandb_run=None
):
    """
    Train model for one epoch.

    Args:
        model: The xAR model to train
        vae: Pretrained AutoencoderKL for encoding images to latent space
        model_params: List of model parameters
        ema_params: List of EMA parameters
        data_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        loss_scaler: Mixed precision loss scaler
        log_writer: TensorBoard log writer
        args: Training arguments
        wandb_run: Wandb run object for logging

    Returns:
        Dictionary of averaged metrics
    """
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Per-iteration learning rate schedule
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Normalize images to [-1, 1] for KL-VAE and encode to latent space
        with torch.no_grad():
            # samples = samples * 2.0 - 1.0  # [0, 1] -> [-1, 1] # already done in VAE
            posterior = vae.encode(samples)
            x = posterior.sample().mul_(0.2325)  # Scale latent to have std=1 (change if using different tokenizer)

        # Forward pass
        with torch.cuda.amp.autocast(dtype=args.amp_dtype):
            loss = model(x, labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Backward pass with gradient scaling
        loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()

        torch.cuda.synchronize()

        # EMA update
        update_ema(ema_params, model_params, rate=args.ema_rate)

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        # Logging
        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

        if wandb_run is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            wandb_run.log({
                'train_loss': loss_value_reduce,
                'lr': lr,
                'epoch': epoch,
                'step': epoch_1000x,
            }, step=epoch_1000x)

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(
        model_without_ddp,
        vae,
        ema_params,
        args,
        epoch,
        batch_size=16,
        log_writer=None,
        cfg=1.0,
        use_ema=True,
        wandb_run=None
):
    """
    Evaluate model by generating sample images.

    Args:
        model_without_ddp: Model without DDP wrapper
        vae: Pretrained AutoencoderKL for decoding latents to images
        ema_params: EMA parameters
        args: Evaluation arguments
        epoch: Current epoch
        batch_size: Batch size for generation
        log_writer: TensorBoard log writer
        cfg: Classifier-free guidance scale
        use_ema: Whether to use EMA parameters
        wandb_run: Wandb run object

    Returns:
        None
    """
    model_without_ddp.eval()

    # Switch to EMA params if requested
    if use_ema:
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        print("Switch to EMA")
        model_without_ddp.load_state_dict(ema_state_dict)

    # Generate random class labels
    class_num = getattr(args, 'class_num', 1000)
    labels_gen = torch.randint(0, class_num, (batch_size,)).cuda()

    amp_dtype = getattr(args, 'amp_dtype', torch.bfloat16)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=amp_dtype):
            sampled_tokens = model_without_ddp.sample_tokens(
                bsz=batch_size,
                num_iter=args.num_iter,
                cfg=cfg,
                cfg_schedule=args.cfg_schedule,
                labels=labels_gen,
                temperature=args.temperature,
            )
            # Decode latents to images via pretrained VAE
            sampled_images = vae.decode(sampled_tokens / 0.2325)

    # # Convert from [-1, 1] to [0, 1] for visualization # already done in VAE decode, so skip this step
    # sampled_images = sampled_images.detach().cpu().clamp(-1, 1) * 0.5 + 0.5
    sampled_images = sampled_images.detach().cpu().clamp(0, 1)

    if misc.get_rank() == 0:
        nrow = math.ceil(math.sqrt(batch_size))
        grid = vutils.make_grid(sampled_images, nrow=nrow, normalize=False, padding=2)

        # Save grid PNG
        save_folder = os.path.join(args.output_dir, "samples")
        os.makedirs(save_folder, exist_ok=True)
        grid_np = np.round(np.clip(grid.permute(1, 2, 0).numpy() * 255, 0, 255)).astype(np.uint8)
        grid_path = os.path.join(save_folder, f"epoch{epoch:04d}_cfg{cfg}.png")

        # Use PIL to save (avoid cv2 dependency issues)
        try:
            from PIL import Image
            Image.fromarray(grid_np).save(grid_path)
        except ImportError:
            import cv2
            cv2.imwrite(grid_path, grid_np[:, :, ::-1])

        print("Saved image grid to:", grid_path)

        # TensorBoard logging
        if log_writer is not None:
            log_writer.add_image("generated", grid, epoch)

        # Wandb logging
        if wandb_run is not None:
            import wandb
            wandb_run.log(
                {"generated_grid": wandb.Image(grid_np, caption=f"epoch {epoch} cfg={cfg}")},
                step=epoch,
            )

    # Switch back from EMA
    if use_ema:
        print("Switch back from EMA")
        model_without_ddp.load_state_dict(model_state_dict)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def generate_samples(
        model,
        vae,
        num_samples,
        batch_size,
        class_num,
        num_steps,
        cfg=1.0,
        amp_dtype=torch.bfloat16,
        save_dir=None
):
    """
    Generate a batch of samples for evaluation.

    Args:
        model: The trained xAR model
        vae: Pretrained AutoencoderKL for decoding latents to images
        num_samples: Total number of samples to generate
        batch_size: Batch size for generation
        class_num: Number of classes
        num_steps: Number of diffusion steps
        cfg: Classifier-free guidance scale
        amp_dtype: Mixed precision dtype
        save_dir: Directory to save samples

    Returns:
        List of generated images
    """
    model.eval()
    all_images = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        labels = torch.randint(0, class_num, (current_batch_size,)).cuda()

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                tokens = model.sample_tokens(
                    bsz=current_batch_size,
                    num_iter=num_steps,
                    cfg=cfg,
                    labels=labels,
                )
                images = vae.decode(tokens / 0.2325)  # Decode latents to images

        # # Convert from [-1, 1] to [0, 1]: already done in VAE decode, so skip this step
        # images = images.detach().cpu().clamp(-1, 1) * 0.5 + 0.5
        images = images.detach().cpu().clamp(0, 1)
        all_images.append(images)

        if batch_idx % 10 == 0:
            print(f"Generated {(batch_idx + 1) * batch_size} / {num_samples} samples")

    all_images = torch.cat(all_images, dim=0)[:num_samples]

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for i, img in enumerate(all_images):
            img_np = np.round(np.clip(img.permute(1, 2, 0).numpy() * 255, 0, 255)).astype(np.uint8)
            try:
                from PIL import Image
                Image.fromarray(img_np).save(os.path.join(save_dir, f"{i:05d}.png"))
            except ImportError:
                import cv2
                cv2.imwrite(os.path.join(save_dir, f"{i:05d}.png"), img_np[:, :, ::-1])

    return all_images

"""
Efficient training engine — zero cross-node reduce during training loop.

All logging uses rank-0 local values only.  The only distributed
primitives that remain are:
  - DDP backward (gradient all-reduce, unavoidable)
  - barrier in evaluate() to sync before/after generation
"""

import math
import sys
import time
import copy
from typing import Iterable

import numpy as np
import os
import torch
import torchvision.utils as vutils

import util.misc as misc
import util.lr_sched as lr_sched


def update_ema(target_params, source_params, rate=0.99):
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
        wandb_run=None,
):
    model.train(True)

    print_freq = 20
    is_rank0 = misc.get_rank() == 0
    use_cached = getattr(args, 'use_cached', False)

    optimizer.zero_grad()

    if is_rank0 and log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # Running stats (rank-0 only, no SmoothedValue / MetricLogger overhead)
    loss_sum = 0.0
    loss_count = 0
    t_start = time.time()
    t_iter = time.time()

    for step, batch in enumerate(data_loader):
        # LR schedule
        lr_sched.adjust_learning_rate(optimizer, step / len(data_loader) + epoch, args)

        # ---------- data ----------
        if use_cached:
            cached_latent, labels, img_pixel = batch[0], batch[1], batch[2]
            feat = batch[3] if len(batch) == 4 else None
            x = cached_latent.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            img_pixel = img_pixel.to(device, non_blocking=True)
            if feat is not None:
                feat = feat.to(device, non_blocking=True)
        else:
            samples, labels = batch
            samples = samples.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            img_pixel = None
            feat = None
            with torch.no_grad():
                x = vae.encode(samples)

        # ---------- forward ----------
        precision_warmup = getattr(args, 'precision_warmup_epochs', 0)
        amp_dtype = torch.float32 if epoch < precision_warmup else args.amp_dtype
        with torch.cuda.amp.autocast(dtype=amp_dtype):
            loss, loss_log = model(x, labels, img_pixel, feat)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # ---------- backward + step ----------
        loss_scaler(loss, optimizer, clip_grad=args.grad_clip,
                    parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()

        # EMA update
        update_ema(ema_params, model_params, rate=args.ema_rate)

        # ---------- rank-0 logging (no reduce) ----------
        loss_sum += loss_value
        loss_count += 1

        if is_rank0 and (step % print_freq == 0 or step == len(data_loader) - 1):
            lr = optimizer.param_groups[0]["lr"]
            t_now = time.time()
            sec_per_iter = (t_now - t_iter) / print_freq if step > 0 else t_now - t_iter
            eta = sec_per_iter * (len(data_loader) - step)
            t_iter = t_now

            extra = "  ".join(f"{k}: {v:.4f}" for k, v in loss_log.items())
            print(
                f"Epoch [{epoch}][{step:>{len(str(len(data_loader)))}d}/{len(data_loader)}]  "
                f"loss: {loss_value:.4f}  {extra}  "
                f"lr: {lr:.6f}  "
                f"iter: {sec_per_iter:.3f}s  eta: {int(eta)}s  "
                f"mem: {torch.cuda.max_memory_allocated() / 1024**2:.0f}MB"
            )

            train_step = epoch * len(data_loader) + step
            epoch_1000x = int((step / len(data_loader) + epoch) * 1000)

            if log_writer is not None:
                log_writer.add_scalar('train_loss', loss_value, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
                log_writer.add_scalar('train_step', train_step, epoch_1000x)
                for k, v in loss_log.items():
                    log_writer.add_scalar(f'train_{k}', v, epoch_1000x)

            if wandb_run is not None:
                wandb_log = {
                    'train_loss': loss_value,
                    'lr': lr,
                    'epoch': epoch,
                    'step': epoch_1000x,
                    'train_step': train_step,
                }
                for k, v in loss_log.items():
                    wandb_log[f'train_{k}'] = v
                wandb_run.log(wandb_log, step=epoch_1000x)

    # Epoch summary (rank-0 local average, no all-reduce)
    avg_loss = loss_sum / loss_count if loss_count else 0.0
    total_time = time.time() - t_start
    if is_rank0:
        print(f"Epoch {epoch} done — avg loss: {avg_loss:.4f}  "
              f"time: {total_time:.1f}s ({total_time / max(loss_count, 1):.3f}s/it)")

    return {"loss": avg_loss}


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
        wandb_run=None,
):
    model_without_ddp.eval()

    if use_ema:
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        print("Switch to EMA")
        model_without_ddp.load_state_dict(ema_state_dict)

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
            sampled_images = vae.decode(sampled_tokens)

    sampled_images = sampled_images.detach().cpu().clamp(0, 1)

    if misc.get_rank() == 0:
        nrow = math.ceil(math.sqrt(batch_size))
        grid = vutils.make_grid(sampled_images, nrow=nrow, normalize=False, padding=2)

        save_folder = os.path.join(args.output_dir, "samples")
        os.makedirs(save_folder, exist_ok=True)
        grid_np = np.round(np.clip(grid.permute(1, 2, 0).numpy() * 255, 0, 255)).astype(np.uint8)
        grid_path = os.path.join(save_folder, f"epoch{epoch:04d}_cfg{cfg}.png")

        try:
            from PIL import Image
            Image.fromarray(grid_np).save(grid_path)
        except ImportError:
            import cv2
            cv2.imwrite(grid_path, grid_np[:, :, ::-1])

        print("Saved image grid to:", grid_path)

        if log_writer is not None:
            log_writer.add_image("generated", grid, epoch)

        if wandb_run is not None:
            import wandb
            wandb_run.log(
                {"generated_grid": wandb.Image(grid_np, caption=f"epoch {epoch} cfg={cfg}")},
                step=epoch,
            )

    if use_ema:
        print("Switch back from EMA")
        model_without_ddp.load_state_dict(model_state_dict)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

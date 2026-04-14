"""
Standalone evaluation script for MAR models.

Loads a checkpoint, generates images across GPUs, computes FID and IS
metrics (via torch_fidelity), and optionally saves samples / logs to CSV.

Usage (multi-GPU):
    torchrun --nproc_per_node=4 eval.py \
        --exps_dir ./ho_mar_0311 --run_name my_run --train_step last

Usage (single GPU):
    python eval.py --exps_dir ./ho_mar_0311 --run_name my_run --train_step 10
"""

import argparse
import copy
import csv
import fcntl
import math
import os
import shutil
import time

import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import yaml

import models as models_mar
import util.misc as misc
import vae as vae_module


# ──────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────

def get_args_parser():
    p = argparse.ArgumentParser("MAR Evaluation", add_help=False)

    # Required
    p.add_argument("--exps_dir", required=True, type=str,
                   help="Experiments root directory")
    p.add_argument("--run_name", required=True, type=str,
                   help="Experiment run name")
    p.add_argument("--train_step", required=True, type=str,
                   help='Checkpoint id, e.g. "10" or "last"')

    # Generation
    p.add_argument("--num_images", default=50000, type=int)
    p.add_argument("--batch_size", default=64, type=int,
                   help="Per-GPU batch size")
    p.add_argument("--num_iter", default=64, type=int,
                   help="Number of autoregressive iterations")
    p.add_argument("--cfg", default=1.0, type=float,
                   help="Classifier-free guidance scale")
    p.add_argument("--cfg_schedule", default="linear", type=str)
    p.add_argument("--temperature", default=1.0, type=float)
    p.add_argument("--dtype", default="bf16", type=str,
                   choices=["fp32", "fp16", "bf16"])
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--no_ema", action="store_true",
                   help="Use training weights instead of EMA weights")

    # Output options
    p.add_argument("--save_samples_dir", default=None, type=str,
                   help="Directory to save generated PNGs")
    p.add_argument("--save_npz", action="store_true",
                   help="Save samples as NPZ (for ADM eval)")
    p.add_argument("--csv_file", default=None, type=str,
                   help="CSV file to append results (with file locking)")
    p.add_argument("--clean_samples", action="store_true",
                   help="Remove generated image directory after evaluation")
    p.add_argument("--fid_stats", default="fid_stats/adm_in256_stats.npz",
                   type=str, help="Path to precomputed FID statistics")

    # Distributed (populated by torchrun)
    p.add_argument("--world_size", default=1, type=int)
    p.add_argument("--local_rank", default=-1, type=int)
    p.add_argument("--dist_on_itp", action="store_true")
    p.add_argument("--dist_url", default="env://")

    return p


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

_DATASET_CLASS_NUM = {
    "imagenet": 1000,
    "tiny-imagenet-hf": 200,
    "cifar10-hf": 10,
    "cifar10": 10,
    "mnist-hf": 10,
    "mnist": 10,
}


def infer_class_num(cfg):
    """Derive class_num from the YAML config."""
    # Prefer explicit value in model_config
    class_num = cfg.get("model_config", {}).get("class_num")
    if class_num is not None:
        return int(class_num)
    # Infer from dataset name
    dataset_name = cfg.get("dataset", {}).get("name", "imagenet")
    return _DATASET_CLASS_NUM.get(dataset_name, 1000)


_CSV_COLUMNS = [
    "run_name", "train_step", "fid", "is",
    "num_images", "batch_size", "num_iter", "cfg", "cfg_schedule",
    "temperature", "dtype", "seed",
    "use_ema", "ema_rate",
]


def log_to_csv(csv_path, args, fid, inception_score, use_ema, ema_rate):
    """Append a result row to *csv_path* with exclusive file locking."""
    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(_CSV_COLUMNS)
            writer.writerow([
                args.run_name, args.train_step,
                f"{fid:.4f}", f"{inception_score:.4f}",
                args.num_images, args.batch_size, args.num_iter, args.cfg, args.cfg_schedule,
                args.temperature, args.dtype, args.seed,
                use_ema, ema_rate if ema_rate is not None else "",
            ])
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main(args):
    # ── paths ─────────────────────────────────────────────────────────
    run_dir = os.path.join(args.exps_dir, args.run_name)
    config_path = os.path.join(run_dir, "config.yaml")
    ckpt_path = os.path.join(run_dir, f"checkpoint-{args.train_step}.pth")

    assert os.path.isfile(config_path), f"Config not found: {config_path}"
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    # ── distributed init ──────────────────────────────────────────────
    misc.init_distributed_mode(args)
    rank = misc.get_rank()
    world_size = misc.get_world_size()

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    cudnn.benchmark = True

    device = torch.device("cuda")
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16,
                 "bf16": torch.bfloat16}
    amp_dtype = dtype_map[args.dtype]

    # ── load config ───────────────────────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    vae_name = cfg["vae"]
    vae_config = cfg["vae_config"]
    model_name = cfg["model"]
    model_config = cfg["model_config"]

    class_num = infer_class_num(cfg)
    model_config["class_num"] = class_num
    img_size = model_config.get("img_size", 256)

    # ── build VAE ─────────────────────────────────────────────────────
    vae_cls = getattr(vae_module, vae_name)
    vae = vae_cls(**vae_config).to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    # ── build model ───────────────────────────────────────────────────
    model_config["grad_checkpointing"] = False
    model = models_mar.__dict__[model_name](**model_config)

    # ── load checkpoint ─────────────────────────────────────────────────
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ema_state = checkpoint.get("model_ema")
    train_state = checkpoint.get("model")

    # Log available weight info
    ema_norm = None
    if ema_state is not None:
        ema_norm = sum(v.float().norm().item() ** 2 for v in ema_state.values()) ** 0.5
    train_norm = None
    if train_state is not None:
        train_norm = sum(v.float().norm().item() ** 2 for v in train_state.values()) ** 0.5

    if rank == 0:
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        print(f"  EMA weights:   {'available' if ema_state is not None else 'NOT available'}")
        print(f"  Train weights: {'available' if train_state is not None else 'NOT available'}")
        if ema_state is not None:
            print(f"  EMA   — #params: {len(ema_state)}, L2 norm: {ema_norm:.4f}")
        if train_state is not None:
            print(f"  Train — #params: {len(train_state)}, L2 norm: {train_norm:.4f}")

    use_ema = not args.no_ema
    if use_ema and ema_state is not None:
        model.load_state_dict(ema_state)
        print(f"Loaded EMA weights from {ckpt_path}")
    elif train_state is not None:
        model.load_state_dict(train_state)
        if use_ema:
            use_ema = False  # reflect actual state
            print(f"WARNING: --no_ema not set but model_ema is None, "
                  f"falling back to training weights from {ckpt_path}")
        else:
            print(f"Loaded training weights from {ckpt_path} (--no_ema)")
    else:
        raise RuntimeError(f"No loadable weights found in {ckpt_path}")
    del checkpoint, ema_state, train_state
    model.to(device).eval()
    torch.cuda.empty_cache()

    # ── prepare save folder ───────────────────────────────────────────
    if args.save_samples_dir:
        save_folder = args.save_samples_dir
    else:
        save_folder = os.path.join(
            run_dir,
            f"eval_tmp_{args.train_step}_{os.getpid()}")
    if rank == 0:
        os.makedirs(save_folder, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    # ── label distribution (balanced per class) ───────────────────────
    num_images = args.num_images
    labels_all = np.arange(0, class_num).repeat(
        math.ceil(num_images / class_num))
    # Pad so every rank has equal work
    total_needed = math.ceil(num_images / (args.batch_size * world_size)) \
                   * args.batch_size * world_size
    if len(labels_all) < total_needed:
        labels_all = np.concatenate(
            [labels_all, np.zeros(total_needed - len(labels_all), dtype=int)])
    labels_all = labels_all[:total_needed]

    batch_size = args.batch_size
    num_steps = total_needed // (batch_size * world_size)

    # ── generation loop ───────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    print(f"[Rank {rank}] Generating {num_images} images "
          f"({num_steps} steps, bs={batch_size}, world_size={world_size})")

    gen_img_cnt = 0
    used_time = 0.0

    pbar = tqdm(range(num_steps), desc="Generating", disable=(rank != 0),
                unit="batch", total=num_steps)
    for i in pbar:
        start_idx = world_size * batch_size * i + rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = torch.from_numpy(
            labels_all[start_idx:end_idx].copy()).long().to(device)
        cur_bsz = labels_gen.shape[0]

        torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                sampled_tokens = model.sample_tokens(
                    bsz=cur_bsz,
                    num_iter=args.num_iter,
                    cfg=args.cfg,
                    cfg_schedule=args.cfg_schedule,
                    labels=labels_gen,
                    temperature=args.temperature,
                )
                sampled_images = vae.decode(sampled_tokens)

        if i >= 1:
            torch.cuda.synchronize()
            used_time += time.time() - t0
            gen_img_cnt += cur_bsz

        sampled_images = sampled_images.detach().cpu().clamp(0, 1)

        # Save PNGs
        for b in range(cur_bsz):
            img_id = i * batch_size * world_size + rank * batch_size + b
            if img_id >= num_images:
                break
            img_np = np.round(
                sampled_images[b].numpy().transpose(1, 2, 0) * 255
            ).clip(0, 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(save_folder, f"{img_id:05d}.png"),
                img_np[:, :, ::-1],  # RGB → BGR
            )

        done = min((i + 1) * batch_size * world_size, num_images)
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        pbar.set_postfix(images=f"{done}/{num_images}", vram=f"{peak_gb:.1f}G")

    if dist.is_initialized():
        dist.barrier()

    if gen_img_cnt > 0 and rank == 0:
        print(f"Generation speed: {used_time / gen_img_cnt:.4f} sec/image "
              f"(excluding first batch warmup)")

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"[Rank {rank}] Peak VRAM: {peak_mem:.2f} GB")

    # ── save NPZ (rank 0) ────────────────────────────────────────────
    if args.save_npz and rank == 0:
        print("Building NPZ …")
        npz_path = os.path.join(run_dir, f"samples_{args.train_step}.npz")
        all_imgs = np.zeros((num_images, img_size, img_size, 3), dtype=np.uint8)
        for idx in range(num_images):
            png_path = os.path.join(save_folder, f"{idx:05d}.png")
            img_bgr = cv2.imread(png_path)
            all_imgs[idx] = img_bgr[:, :, ::-1]  # BGR → RGB
        np.savez(npz_path, arr_0=all_imgs)
        print(f"Saved NPZ ({num_images} images) to {npz_path}")

    # ── compute FID / IS (rank 0) ────────────────────────────────────
    fid, inception_score = None, None
    if rank == 0:
        import torch_fidelity
        print("Computing FID and IS …")
        metrics = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=args.fid_stats,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics["frechet_inception_distance"]
        inception_score = metrics["inception_score_mean"]
        print(f"FID: {fid:.4f}, IS: {inception_score:.4f}")

    # ── log to CSV (rank 0) ──────────────────────────────────────────
    if rank == 0 and args.csv_file and fid is not None:
        ema_rate = cfg.get("training", {}).get("ema_rate")
        log_to_csv(args.csv_file, args, fid, inception_score, use_ema, ema_rate)
        print(f"Results appended to {args.csv_file}")

    # ── cleanup generated image folder ─────────────────────────────────
    if rank == 0 and (args.clean_samples or not args.save_samples_dir):
        shutil.rmtree(save_folder, ignore_errors=True)
        print(f"Removed image folder {save_folder}")

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)

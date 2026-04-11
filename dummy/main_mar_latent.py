"""
Main training script for xAR Latent models.

This script trains xAR models in latent space using a pretrained KL-VAE for image tokenization.
It supports multiple datasets, wandb logging, and distributed training.

Usage:
    python main_mar_latent.py --config configs/marflow_ssl_latent.yaml \
        --vae_ckpt /path/to/kl16.ckpt

    # With wandb logging
    python main_mar_latent.py --config configs/marflow_ssl_latent.yaml \
        --vae_ckpt /path/to/kl16.ckpt \
        --wandb_project xar_latent --wandb_entity your_entity
"""

import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import yaml
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.crop import center_crop_arr
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models.vae import AutoencoderKL
import models as models_mar
from engine_mar_latent import train_one_epoch, evaluate
import copy


class SingleSampleDataset(torch.utils.data.Dataset):
    """Debug wrapper: repeats a single fixed sample N times."""

    def __init__(self, sample, length=1024):
        """
        Args:
            sample: tuple of (image_tensor, label) - the fixed sample to repeat
            length: number of times to repeat the sample
        """
        self.sample = sample
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.sample


class HFImageDataset(torch.utils.data.Dataset):
    """Thin wrapper around a Hugging Face image classification dataset.

    Handles different image field names across datasets:
      - 'image': zh-plus/tiny-imagenet and most HF vision datasets
      - 'img':   uoft-cs-pytorch/cifar10

    Usage:
        dataset_train = HFImageDataset("zh-plus/tiny-imagenet", split="train", transform=...)
        dataset_train = HFImageDataset("uoft-cs-pytorch/cifar10", split="train", transform=...)
    """

    def __init__(self, hf_path: str, split: str, transform=None):
        from datasets import load_dataset
        self.ds = load_dataset(hf_path, split=split)
        self.transform = transform

        # Detect image field name once at init
        sample = self.ds[0]
        if 'image' in sample:
            self.img_key = 'image'
        elif 'img' in sample:
            self.img_key = 'img'
        else:
            raise KeyError(f"No 'image' or 'img' field found in dataset. Keys: {list(sample.keys())}")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample[self.img_key].convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, sample['label']


def get_args_parser():
    parser = argparse.ArgumentParser('xAR Latent Training', add_help=False)

    # Config
    parser.add_argument('--config', required=True, type=str,
                        help='Path to YAML model config (e.g. configs/marflow_ssl_latent.yaml)')

    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus)')
    parser.add_argument('--epochs', default=400, type=int)

    # Pretrained VAE settings
    parser.add_argument('--vae_ckpt', required=True, type=str,
                        help='Path to pretrained KL-VAE checkpoint')
    parser.add_argument('--vae_embed_dim', default=16, type=int,
                        help='Latent channels of the pretrained VAE')
    parser.add_argument('--vae_ch_mult', default=[1, 1, 2, 2, 4], type=int, nargs='+',
                        help='Channel multipliers for the pretrained VAE')
    parser.add_argument('--vae_use_variational', action='store_true', default=True,
                        help='Use variational (KL) mode for VAE')
    parser.add_argument('--vae_deterministic', action='store_false', dest='vae_use_variational',
                        help='Use deterministic mode for VAE')

    # Generation parameters
    parser.add_argument('--num_iter', default=50, type=int,
                        help='Number of autoregressive iterations to generate an image')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance scale')
    parser.add_argument('--cfg_schedule', default="linear", type=str)
    parser.add_argument('--temperature', default=1.0, type=float,
                        help='Sampling temperature')
    parser.add_argument('--eval_freq', type=int, default=40)
    parser.add_argument('--save_last_freq', type=int, default=5)
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--eval_bsz', type=int, default=64)

    # Optimizer
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--grad_checkpointing', action='store_true',
                        help='Override grad_checkpointing in model config')
    parser.add_argument('--lr', type=float, default=None, metavar='LR')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='Base lr: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR')
    parser.add_argument('--lr_schedule', type=str, default='constant')
    parser.add_argument('--warmup_epochs', type=int, default=100, metavar='N')
    parser.add_argument('--ema_rate', default=0.9999, type=float)
    parser.add_argument('--grad_clip', type=float, default=3.0)

    parser.add_argument('--run_name', default="exp1", type=str)
    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='bf16', type=str, choices=['fp32', 'fp16', 'bf16'],
                        help='Training precision. bf16: no loss scaler needed; fp16: uses GradScaler.')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--debug_one_image', action='store_true',
                        help='Debug mode: dataset returns a single sample repeated N times')
    parser.add_argument('--resume', default='',
                        help='Path to a checkpoint directory (loads checkpoint-last.pth inside it)')
    parser.add_argument('--resume_last', action='store_true',
                        help='Automatically resume from checkpoint-last.pth in the current run dir')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Wandb
    parser.add_argument('--wandb_key', default='', type=str)
    parser.add_argument('--wandb_entity', default='', type=str)
    parser.add_argument('--wandb_project', default='', type=str)

    # Distributed
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    return parser


def main(args):
    # ------------------------------------------------------------------
    # Resolve run directory: output_dir/run_name
    # ------------------------------------------------------------------
    run_dir = os.path.join(args.output_dir, args.run_name)
    args.output_dir = run_dir
    args.log_dir = run_dir
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    if args.resume_last and not args.resume:
        args.resume = run_dir

    # ------------------------------------------------------------------
    # Load model config from YAML
    # ------------------------------------------------------------------
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg['model']
    model_config = cfg['model_config']

    # Dataset config from YAML
    dataset_cfg = cfg['dataset']
    args.dataset = dataset_cfg['name']
    args.data_path = dataset_cfg.get('path', './data')

    # Training config from YAML (overrides CLI defaults, but CLI explicit values win)
    training_cfg = cfg.get('training', {})
    _float_keys = {'lr', 'blr', 'min_lr', 'weight_decay', 'ema_rate', 'grad_clip'}
    for key, value in training_cfg.items():
        if hasattr(args, key):
            if key in _float_keys:
                value = float(value)
            setattr(args, key, value)

    # CLI --grad_checkpointing flag overrides the YAML value
    if args.grad_checkpointing:
        model_config['grad_checkpointing'] = True

    # Reflect img_size from model config so the data pipeline uses the same value
    args.img_size = model_config.get('img_size', 256)

    # ------------------------------------------------------------------
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    print(f"Model: {model_name}")
    print(f"Model config: {model_config}")

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # ------------------------------------------------------------------
    # Wandb initialization
    # ------------------------------------------------------------------
    wandb_run = None
    if global_rank == 0 and args.wandb_project:
        import wandb
        if args.wandb_key:
            wandb.login(key=args.wandb_key)
        wandb_run = wandb.init(
            entity=args.wandb_entity if args.wandb_entity else None,
            project=args.wandb_project,
            name=args.run_name if args.run_name else None,
            config={**vars(args), 'model': model_name, **model_config},
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    transform_train = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    if args.dataset == 'tiny-imagenet-hf':
        dataset_train = HFImageDataset("zh-plus/tiny-imagenet", split="train", transform=transform_train)
        args.class_num = 200
    elif args.dataset == 'cifar10-hf':
        dataset_train = HFImageDataset("uoft-cs/cifar10", split="train", transform=transform_train)
        args.class_num = 10
    elif args.dataset == 'mnist-hf':
        transform_train = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
            transforms.ToTensor(),
        ])
        dataset_train = HFImageDataset("ylecun/mnist", split="train", transform=transform_train)
        args.class_num = 10
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
        args.class_num = 10
    elif args.dataset == 'mnist':
        # MNIST is grayscale, convert to RGB
        transform_mnist = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        dataset_train = datasets.MNIST(root=args.data_path, train=True, download=True, transform=transform_mnist)
        args.class_num = 10
    else:
        print(f"Dataset {args.dataset}. Using ImageFolder with path: {args.data_path}")
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
        args.class_num = 1000

    if args.debug_one_image:
        args.batch_size = 32
        args.lr = 1e-4
        model_config['grad_checkpointing'] = False
        args.eval_bsz = 4
        # Get and fix a single sample (avoid random transform variations)
        # Only rank 0 loads the image, then broadcast to all GPUs for consistency
        if global_rank == 0:
            debug_img, debug_label = dataset_train[200]
            debug_label = torch.tensor(debug_label)
        else:
            debug_img = torch.zeros(3, args.img_size, args.img_size)
            debug_label = torch.tensor(0)

        # Broadcast to ensure all devices have the same image
        if num_tasks > 1:
            debug_img = debug_img.cuda()
            debug_label = debug_label.cuda()
            torch.distributed.broadcast(debug_img, src=0)
            torch.distributed.broadcast(debug_label, src=0)
            debug_img = debug_img.cpu()
            debug_label = debug_label.cpu().item()
        print(f"[DEBUG] Image shape: {debug_img.shape}, target img_size: {args.img_size}")
        # Resize if needed
        if debug_img.shape[1] != args.img_size or debug_img.shape[2] != args.img_size:
            debug_img = torch.nn.functional.interpolate(
                debug_img.unsqueeze(0), size=(args.img_size, args.img_size), mode='bilinear', align_corners=False
            ).squeeze(0)
            print(f"[DEBUG] Resized to: {debug_img.shape}")
        # Save the debug image
        debug_img_np = (debug_img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        debug_save_path = os.path.join(run_dir, "debug_train_image.png")
        try:
            from PIL import Image
            Image.fromarray(debug_img_np).save(debug_save_path)
        except ImportError:
            import cv2
            cv2.imwrite(debug_save_path, debug_img_np[:, :, ::-1])
        print(f"[DEBUG] Saved training image to: {debug_save_path} (label={debug_label})")
        # Use fixed sample (no random transform variations)
        # length = batch_size * num_tasks * iterations_per_epoch
        dataset_train = SingleSampleDataset((debug_img, debug_label), length=args.batch_size * num_tasks * 100)
        print("[DEBUG] debug_one_image: dataset replaced with fixed single sample")

    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # ------------------------------------------------------------------
    # Save GT image grid for visualization
    # ------------------------------------------------------------------
    if global_rank == 0:
        from torchvision.utils import save_image, make_grid
        # Get a batch of training images
        sample_iter = iter(data_loader_train)
        sample_images, sample_labels = next(sample_iter)
        # Create grid (8 images per row)
        nrow = min(8, sample_images.size(0))
        grid = make_grid(sample_images, nrow=nrow, normalize=False, padding=2)
        # Save the grid
        gt_grid_path = os.path.join(run_dir, "gt_train_images_grid.png")
        save_image(grid, gt_grid_path)
        print(f"Saved GT training image grid to: {gt_grid_path}")
        # Log to wandb if available
        if wandb_run is not None:
            import wandb
            wandb_run.log({"gt_train_images": wandb.Image(gt_grid_path)}, step=0)
        del sample_iter, sample_images, sample_labels

    # ------------------------------------------------------------------
    # Pretrained KL-VAE
    # ------------------------------------------------------------------
    vae = AutoencoderKL(
        embed_dim=args.vae_embed_dim,
        ch_mult=tuple(args.vae_ch_mult),
        use_variational=args.vae_use_variational,
        ckpt_path=args.vae_ckpt,
    ).cuda().eval()
    for param in vae.parameters():
        param.requires_grad = False

    print(f"AutoencoderKL: embed_dim={args.vae_embed_dim}, ch_mult={args.vae_ch_mult}, "
          f"variational={args.vae_use_variational}, ckpt={args.vae_ckpt}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_config['class_num'] = args.class_num
    model = models_mar.__dict__[model_name](**model_config)

    print("Model = %s" % str(model))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))

    n_params_diffloss = sum(p.numel() for p in model.diffloss.parameters() if p.requires_grad)
    print("number of diffloss parameters: {}M".format(n_params_diffloss / 1e6))

    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * misc.get_world_size()

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # Mixed precision setup
    _dtype_map = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
    args.amp_dtype = _dtype_map[args.dtype]
    loss_scaler = NativeScaler()

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    if args.resume and os.path.exists(os.path.join(args.resume, "checkpoint-last.pth")):
        checkpoint = torch.load(os.path.join(args.resume, "checkpoint-last.pth"), map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        model_params = list(model_without_ddp.parameters())
        ema_state_dict = checkpoint['model_ema']
        ema_params = [ema_state_dict[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resume checkpoint %s" % args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")
        del checkpoint
    else:
        model_params = list(model_without_ddp.parameters())
        ema_params = copy.deepcopy(model_params)
        print("Training from scratch")

    # ------------------------------------------------------------------
    # Evaluation only mode
    # ------------------------------------------------------------------
    if args.evaluate:
        torch.cuda.empty_cache()
        evaluate(model_without_ddp, vae, ema_params, args, 0, batch_size=args.eval_bsz,
                 log_writer=log_writer, cfg=args.cfg, use_ema=True, wandb_run=wandb_run)
        if wandb_run is not None:
            wandb_run.finish()
        return

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            model, vae,
            model_params, ema_params,
            data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,
            wandb_run=wandb_run,
        )

        # Save checkpoint
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                ema_params=ema_params, epoch_name="last"
            )

        # Online evaluation
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=args.eval_bsz,
                     log_writer=log_writer, cfg=1.0, use_ema=False, wandb_run=wandb_run)

            if not (args.cfg == 1.0 or args.cfg == 0.0):
                evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=args.eval_bsz // 2,
                         log_writer=log_writer, cfg=args.cfg, use_ema=False, wandb_run=wandb_run)
            torch.cuda.empty_cache()

        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

    total_time = time.time() - start_time
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

import argparse
import datetime
import numpy as np
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import util.misc as misc
from util.loader import ImageFolderWithFilename
from util.crop import center_crop_arr

from typing import Iterable


def get_args_parser():
    parser = argparse.ArgumentParser('Cache DINOv2 features', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')

    # DINOv2 parameters
    parser.add_argument('--dinov2_repo', default='facebookresearch/dinov2', type=str,
                        help='torch.hub repo for DINOv2')
    parser.add_argument('--dinov2_name', default='dinov2_vitg14_reg', type=str,
                        help='DINOv2 model name to load via torch.hub')
    parser.add_argument('--dinov2_input_size', default=224, type=int,
                        help='Image input size fed to DINOv2 (must be divisible by patch_size)')
    parser.add_argument('--save_dtype', default='fp16', choices=['fp16', 'fp32'],
                        help='dtype used when writing features to disk')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # caching features
    parser.add_argument('--cached_path', default='', help='path to cached features')

    return parser


def cache_features(model,
                   data_loader: Iterable,
                   device: torch.device,
                   args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Caching: '
    print_freq = 20

    save_dtype = torch.float16 if args.save_dtype == 'fp16' else torch.float32

    def extract(x):
        t = model.prepare_tokens_with_masks(x)
        for blk in model.blocks:
            t = blk(t)
        return model.norm(t)

    for samples, _, paths in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device, non_blocking=True)

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            feat = extract(samples)
            feat_flip = extract(samples.flip(dims=[3]))

        feat_np = feat.to(save_dtype).cpu().numpy()
        feat_flip_np = feat_flip.to(save_dtype).cpu().numpy()

        for i, path in enumerate(paths):
            save_path = os.path.join(args.cached_path, path + '.npz')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, feat=feat_np[i], feat_flip=feat_flip_np[i])

        if misc.is_dist_avail_and_initialized():
            torch.cuda.synchronize()

    return


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # DINOv2 expects ImageNet normalization, not [-1, 1].
    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.dinov2_input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_train = ImageFolderWithFilename(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False,
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,  # Don't drop in cache
    )

    # define the dinov2 model
    model = torch.hub.load(args.dinov2_repo, args.dinov2_name, pretrained=True)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    assert args.dinov2_input_size % model.patch_size == 0, (
        f"dinov2_input_size {args.dinov2_input_size} not divisible by "
        f"patch_size {model.patch_size}"
    )

    print(f"Start caching DINOv2 features ({args.dinov2_name} @ {args.dinov2_input_size})")
    start_time = time.time()
    cache_features(
        model,
        data_loader_train,
        device,
        args=args,
    )
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Caching time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

"""
Convert checkpoint from sslenc_class_embed_per_token=True to False.

class_emb.weight: [1000, 64*768] -> [1000, 768]  (mean over 64 tokens)
Optimizer exp_avg / exp_avg_sq for that param: same reshape + mean.
"""
import torch
import copy
import argparse

def convert_class_emb(weight, num_tokens=64, embed_dim=768):
    """[num_classes, num_tokens * embed_dim] -> [num_classes, embed_dim] via mean."""
    return weight.view(weight.shape[0], num_tokens, embed_dim).mean(dim=1)

def find_optimizer_state_idx(optimizer_state, target_shape):
    """Find the optimizer state index whose exp_avg matches target_shape."""
    for idx, state in optimizer_state['state'].items():
        if 'exp_avg' in state and state['exp_avg'].shape == target_shape:
            return idx
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help='Source checkpoint path')
    parser.add_argument('--dst', type=str, required=True, help='Destination checkpoint path')
    parser.add_argument('--num_tokens', type=int, default=64, help='sslenc_class_embed_num')
    parser.add_argument('--embed_dim', type=int, default=768, help='encoder_embed_dim')
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.src}")
    ckpt = torch.load(args.src, map_location='cpu', weights_only=False)

    old_shape = ckpt['model']['class_emb.weight'].shape
    assert old_shape[1] == args.num_tokens * args.embed_dim, \
        f"Expected class_emb.weight dim1 = {args.num_tokens * args.embed_dim}, got {old_shape[1]}"
    print(f"  class_emb.weight: {old_shape} -> [{old_shape[0]}, {args.embed_dim}]")

    # Convert model state dict
    ckpt['model']['class_emb.weight'] = convert_class_emb(
        ckpt['model']['class_emb.weight'], args.num_tokens, args.embed_dim)

    # Convert EMA state dict
    if 'model_ema' in ckpt and 'class_emb.weight' in ckpt['model_ema']:
        ckpt['model_ema']['class_emb.weight'] = convert_class_emb(
            ckpt['model_ema']['class_emb.weight'], args.num_tokens, args.embed_dim)
        print("  Converted model_ema class_emb.weight")

    # Convert optimizer state
    if 'optimizer' in ckpt:
        opt_idx = find_optimizer_state_idx(ckpt['optimizer'], old_shape)
        if opt_idx is not None:
            state = ckpt['optimizer']['state'][opt_idx]
            for key in ['exp_avg', 'exp_avg_sq']:
                if key in state:
                    state[key] = convert_class_emb(state[key], args.num_tokens, args.embed_dim)
            print(f"  Converted optimizer state idx={opt_idx} (exp_avg, exp_avg_sq)")
        else:
            print("  WARNING: Could not find optimizer state for class_emb.weight")

    print(f"Saving converted checkpoint to {args.dst}")
    torch.save(ckpt, args.dst)
    print("Done.")

if __name__ == '__main__':
    main()

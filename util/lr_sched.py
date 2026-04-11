import math


def adjust_learning_rate(optimizer, epoch, args):
    """Per-group warmup with constant/cosine schedule after warmup.

    Each param group may override ``warmup_epochs`` (falls back to
    ``args.warmup_epochs``). ``lr_scale`` is a multiplicative factor applied
    after scheduling. Groups with ``warmup_epochs=0`` start at full LR
    immediately.
    """
    last_lr = None
    for param_group in optimizer.param_groups:
        warmup = param_group.get("warmup_epochs", args.warmup_epochs)
        if epoch < warmup:
            lr = args.lr * epoch / max(1, warmup)
        elif args.lr_schedule == "constant":
            lr = args.lr
        elif args.lr_schedule == "cosine":
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
                1. + math.cos(math.pi * (epoch - warmup) / max(1, args.epochs - warmup))
            )
        else:
            raise NotImplementedError
        scale = param_group.get("lr_scale", 1.0)
        param_group["lr"] = lr * scale
        last_lr = param_group["lr"]
    return last_lr

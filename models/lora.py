import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank update.

    forward(x) = base(x) + (alpha / rank) * (dropout(x) @ A^T) @ B^T
    where A is (rank, in_features), B is (out_features, rank).
    Base weight and bias are frozen; only lora_A and lora_B train.
    lora_B is initialized to zero so the wrapped layer initially matches base(x).
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        assert rank > 0, "LoRA rank must be positive"
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank

        for p in self.base.parameters():
            p.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + self.scaling * lora_out


def wrap_linear_with_lora(parent: nn.Module, dotted_name: str,
                          rank: int, alpha: float, dropout: float = 0.0) -> None:
    """In-place replace `parent.<dotted_name>` (an nn.Linear) with LoRALinear."""
    parts = dotted_name.split(".")
    obj = parent
    for p in parts[:-1]:
        obj = getattr(obj, p)
    leaf = parts[-1]
    base = getattr(obj, leaf)
    if not isinstance(base, nn.Linear):
        raise TypeError(f"{dotted_name} is {type(base).__name__}, expected nn.Linear")
    setattr(obj, leaf, LoRALinear(base, rank, alpha, dropout))

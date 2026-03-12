import math
import torch
from torch import nn
import torch.nn.functional as F

def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def iter_named_trainable_params(module: nn.Module):
    for n, p in module.named_parameters():
        if p.requires_grad:
            yield n, p

def set_ssl_encoder_mode(model, mode: str,
                             lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0,
                             lora_targets=("qkv", "proj", "fc1", "fc2")):
    """
    mode: "freeze" | "full" | "lora"
    """
    mode = mode.lower()
    if mode not in {"freeze", "full", "lora"}:
        raise ValueError(f"Unknown mode: {mode}")

    if mode == "freeze":
        set_requires_grad(model, False)

    elif mode == "full":
        set_requires_grad(model, True)

    elif mode == "lora":
        # freeze encoder
        set_requires_grad(model, False)
        # inject LoRA into selected Linear layers
        apply_lora_to_vit(model, target_linear_names=lora_targets,
                            r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

        # ensure LoRA params are trainable (base weights remain frozen)
        for n, p in model.named_parameters():
            if "lora_A" in n or "lora_B" in n:
                p.requires_grad = True
                
def apply_lora_to_vit(module: nn.Module, target_linear_names=("qkv", "proj", "fc1", "fc2"),
                      r=8, alpha=16, dropout=0.0):
    """
    Replaces selected nn.Linear layers with LoRALinear in-place.
    """
    for name, child in list(module.named_children()):
        # recurse first
        apply_lora_to_vit(child, target_linear_names, r, alpha, dropout)

        # then replace if this is a target Linear
        if isinstance(child, nn.Linear) and any(t in name for t in target_linear_names):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with LoRA.
    W is frozen; LoRA learns A,B such that: y = xW^T + alpha/r * x(B A)^T
    """
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_f = base.in_features
        out_f = base.out_features

        # LoRA params
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))

        # init: A ~ Kaiming, B = 0 (common LoRA init)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        y = self.base(x)
        lora = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return y + self.scaling * lora

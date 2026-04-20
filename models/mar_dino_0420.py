from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import Block

from models.diffloss import DiffLoss
from models.flowloss import FlowLoss


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking


class MAR_DINO_0420(nn.Module):
    """MAR_DINO variant with VPT-Deep style multi-stage class conditioning.

    Instead of a single class_emb buffer that rides through every transformer
    layer, this model allows multiple class_emb buffers, each active over a
    configurable range of layers. The encoder, dinov2, and decoder blocks are
    treated as one 0..N-1 sequence where
        N = encoder_depth + len(dinov2_backbone.blocks) + decoder_depth
    and `class_emb_layers = [[s0, e0], [s1, e1], ...]` defines, for each
    class_emb, the half-open layer range [s, e) during which its buffer is
    injected. Each interval owns its own nn.Embedding, positional embedding,
    and fake_latent (for CFG).

    Intervals may not cross the encoder/dinov2/decoder region boundaries, so
    that every class_emb has a single well-defined embedding dim.
    """
    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=0, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 vae_embed_dim=16,
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 diffloss_class="DiffLoss",
                 diffloss_kwargs={
                     "width": 1024,
                     "depth": 3,
                     "num_sampling_steps": '100',
                     "grad_checkpointing": False,
                 },
                 dinov2_name='dinov2_vitb14_reg',
                 dinov2_repo_path='facebookresearch/dinov2',
                 dinov2_pretrained=True,
                 dinov2_external_pos_embed=False,
                 dino_attn_fp32=False,
                 diffloss_fp32=False,
                 freeze_dino=['patch_embed'],
                 freeze_dino_blocks=[],
                 lora_dino_blocks=[],
                 lora_config=None,
                 use_repa=False,
                 use_repa_cached_feat=False,
                 repa_loss_weight=0.5,
                 repa_save_vram=False,
                 repa_on_unmasked=False,
                 dinov2_repa_name=None,
                 repa_input_size=None,
                 use_align_dino_embed=False,
                 align_dino_embed_loss_weight=0.1,
                 align_dino_embed_loss_type='mse',
                 replace_ls_with_identity=False,
                 class_emb_layers=None,
                 ):
        super().__init__()

        if dinov2_external_pos_embed:
            assert 'pos_embed' in (freeze_dino or []), (
                "dinov2_external_pos_embed=True requires 'pos_embed' in "
                "freeze_dino (dinov2's own pos_embed is no longer used)."
            )

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.grad_checkpointing = grad_checkpointing

        # --------------------------------------------------------------------------
        # Dinov2 backbone (loaded from local repo via torch.hub).
        self.dinov2_name = dinov2_name
        self.dinov2_pretrained = dinov2_pretrained
        self.dinov2_backbone = torch.hub.load(
            dinov2_repo_path, dinov2_name, pretrained=dinov2_pretrained,
        )
        for blk in self.dinov2_backbone.blocks:
            blk.attn.attn_drop = attn_dropout
            blk.attn.proj_drop = nn.Dropout(proj_dropout)
            blk.mlp.drop = nn.Dropout(proj_dropout)

        dino_embed_dim = self.dinov2_backbone.embed_dim
        num_register = self.dinov2_backbone.num_register_tokens
        self.dino_embed_dim = dino_embed_dim
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.num_register = num_register
        self.dinov2_depth = len(self.dinov2_backbone.blocks)
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.total_layers = self.encoder_depth + self.dinov2_depth + self.decoder_depth

        # --------------------------------------------------------------------------
        # Class embedding layer ranges. If None, default to one interval per
        # non-empty region (reproduces the original single-buffer-throughout
        # behavior at the region level, though with separate class_embs).
        if class_emb_layers is None:
            class_emb_layers = []
            if encoder_depth > 0:
                class_emb_layers.append([0, encoder_depth])
            if self.dinov2_depth > 0:
                class_emb_layers.append(
                    [encoder_depth, encoder_depth + self.dinov2_depth]
                )
            if decoder_depth > 0:
                class_emb_layers.append(
                    [encoder_depth + self.dinov2_depth, self.total_layers]
                )
        self._validate_class_emb_layers(class_emb_layers)
        self.class_emb_layers = [list(iv) for iv in class_emb_layers]

        # Per-interval dim + region tag.
        self.class_emb_dims = []
        self.class_emb_regions = []
        for (s, _e) in self.class_emb_layers:
            if s < encoder_depth:
                self.class_emb_dims.append(encoder_embed_dim)
                self.class_emb_regions.append('enc')
            elif s < encoder_depth + self.dinov2_depth:
                self.class_emb_dims.append(dino_embed_dim)
                self.class_emb_regions.append('dino')
            else:
                self.class_emb_dims.append(decoder_embed_dim)
                self.class_emb_regions.append('dec')

        # --------------------------------------------------------------------------
        # Class embedding: one nn.Embedding + pos_emb + fake_latent per interval.
        self.num_classes = class_num
        self.label_drop_prob = label_drop_prob
        self.buffer_size = buffer_size
        self.class_embs = nn.ModuleList([
            nn.Embedding(class_num, d) for d in self.class_emb_dims
        ])
        self.class_pos_embs = nn.ParameterList([
            nn.Parameter(torch.zeros(1, buffer_size, d))
            for d in self.class_emb_dims
        ])
        self.fake_latents = nn.ParameterList([
            nn.Parameter(torch.zeros(1, d)) for d in self.class_emb_dims
        ])

        # --------------------------------------------------------------------------
        # MAR variant masking ratio.
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # MAR encoder specifics. Pos embed only covers the L image tokens;
        # buffer positions are handled per-interval via class_pos_embs.
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)

        self.encoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len, encoder_embed_dim)
        )

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # Dino embedding: projection from encoder_embed_dim to dino_embed_dim for
        # image tokens. Buffer tokens in the dino region are initialized directly
        # at dino_embed_dim via class_embs[i], so no separate buffer projection
        # is needed.
        self.dino_embed = nn.Linear(encoder_embed_dim, dino_embed_dim, bias=True)

        # --------------------------------------------------------------------------
        # Optional external pos_embed for the dinov2 stage.
        self.dinov2_external_pos_embed = dinov2_external_pos_embed
        if dinov2_external_pos_embed:
            self.dinov2_external_pos_embed_param = nn.Parameter(
                torch.zeros(1, 1 + self.seq_len, dino_embed_dim)
            )

        # --------------------------------------------------------------------------
        # MAR decoder specifics. Pos embed covers cls + registers + image tokens;
        # buffer positions are handled per-interval.
        self.decoder_embed = nn.Linear(dino_embed_dim, decoder_embed_dim, bias=True)

        if decoder_depth > 0:
            self.decoder_pos_embed_learned = nn.Parameter(
                torch.zeros(1, 1 + num_register + self.seq_len, decoder_embed_dim)
            )

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        diffloss_kwargs.update({
            "target_channels": self.token_embed_dim,
            "z_channels": decoder_embed_dim,
        })
        diffloss_class = eval(diffloss_class)
        self.diffloss = diffloss_class(**diffloss_kwargs)
        self.diffusion_batch_mul = diffusion_batch_mul
        self.diffloss_fp32 = diffloss_fp32

        self._apply_dino_tuning(
            freeze_dino=freeze_dino,
            freeze_dino_blocks=freeze_dino_blocks,
            lora_dino_blocks=lora_dino_blocks,
            lora_config=lora_config,
        )

        if dino_attn_fp32:
            self._dino_attn_fp32()

        # --------------------------------------------------------------------------
        # REPA
        self.use_repa = use_repa
        self.use_repa_cached_feat = use_repa_cached_feat
        self.repa_loss_weight = float(repa_loss_weight)
        self.repa_save_vram = repa_save_vram
        self.repa_on_unmasked = repa_on_unmasked
        self.repa_feat_pred = None

        # --------------------------------------------------------------------------
        # Align dino_embed output with dinov2 patch_embed on raw pixels.
        self.use_align_dino_embed = use_align_dino_embed
        self.align_dino_embed_loss_weight = float(align_dino_embed_loss_weight)
        if align_dino_embed_loss_type not in ('mse', 'cos'):
            raise ValueError(
                f"align_dino_embed_loss_type must be 'mse' or 'cos', "
                f"got {align_dino_embed_loss_type!r}"
            )
        self.align_dino_embed_loss_type = align_dino_embed_loss_type
        self.align_feat_pred = None
        if use_align_dino_embed and not dinov2_pretrained:
            raise ValueError(
                "use_align_dino_embed=True requires dinov2_pretrained=True "
                "so patch_embed produces a meaningful target."
            )

        if use_repa or use_align_dino_embed:
            self.register_buffer(
                "repa_mean",
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            )
            self.register_buffer(
                "repa_std",
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            )

        if use_repa and not use_repa_cached_feat:
            repa_name = dinov2_repa_name or dinov2_name
            dinov2_repa = torch.hub.load(
                dinov2_repo_path, repa_name, pretrained=True,
            )
            for p in dinov2_repa.parameters():
                p.requires_grad_(False)
            dinov2_repa.eval()

            if dinov2_repa.embed_dim != self.dino_embed_dim:
                raise ValueError(
                    f"REPA dim mismatch: dinov2_repa.embed_dim="
                    f"{dinov2_repa.embed_dim} vs dino_embed_dim="
                    f"{self.dino_embed_dim}. Use the same variant or add a projector."
                )

            if repa_input_size is None:
                repa_input_size = self.seq_w * dinov2_repa.patch_size
            self.repa_input_size = int(repa_input_size)

            repa_patches = (self.repa_input_size // dinov2_repa.patch_size) ** 2
            if repa_patches != self.seq_len:
                raise ValueError(
                    f"REPA patch count mismatch: {repa_patches} (from "
                    f"{self.repa_input_size}/{dinov2_repa.patch_size}) vs "
                    f"seq_len={self.seq_len}. Pick repa_input_size = seq_w * "
                    f"dinov2_repa.patch_size, or use a different dinov2_repa."
                )

            self.dinov2_repa_bank = (dinov2_repa, )
            if repa_save_vram:
                dinov2_repa.cpu()

        self.replace_ls_with_identity = replace_ls_with_identity
        if replace_ls_with_identity:
            for blk in self.dinov2_backbone.blocks:
                blk.ls1 = nn.Identity()
                blk.ls2 = nn.Identity()

    def _validate_class_emb_layers(self, class_emb_layers):
        """Validate the intervals: non-overlapping, sorted, in-range, and each
        entirely within one region (encoder/dinov2/decoder)."""
        E = self.encoder_depth
        D = self.dinov2_depth
        M = self.decoder_depth
        N = E + D + M
        boundaries = [b for b in (E, E + D) if 0 < b < N]

        prev_e = 0
        for idx, iv in enumerate(class_emb_layers):
            if len(iv) != 2:
                raise ValueError(
                    f"class_emb_layers[{idx}] must be [start, end], got {iv}"
                )
            s, e = iv
            if not (isinstance(s, int) and isinstance(e, int)):
                raise ValueError(
                    f"class_emb_layers[{idx}] must contain ints, got {iv}"
                )
            if not (0 <= s < e <= N):
                raise ValueError(
                    f"class_emb_layers[{idx}]=[{s},{e}] out of range "
                    f"[0, {N}] or not strictly ordered"
                )
            if s < prev_e:
                raise ValueError(
                    f"class_emb_layers[{idx}]=[{s},{e}] overlaps previous "
                    f"(prev_end={prev_e})"
                )
            for b in boundaries:
                if s < b < e:
                    raise ValueError(
                        f"class_emb_layers[{idx}]=[{s},{e}] crosses region "
                        f"boundary at layer {b}; intervals must stay within "
                        f"a single region"
                    )
            prev_e = e

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "use_repa", False) and not getattr(self, "use_repa_cached_feat", False):
            self.dinov2_repa_bank[0].eval()
        return self

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        bank = self.__dict__.get('dinov2_repa_bank', None)
        if bank is not None:
            bank[0]._apply(fn, *args, **kwargs)
        return self

    def _apply_dino_tuning(self, freeze_dino=None, freeze_dino_blocks=None,
                           lora_dino_blocks=None, lora_config=None):
        freeze_dino = list(freeze_dino or [])
        freeze_dino_blocks = list(freeze_dino_blocks or [])
        lora_dino_blocks = list(lora_dino_blocks or [])

        for name in freeze_dino:
            if not hasattr(self.dinov2_backbone, name):
                raise ValueError(f"dinov2_backbone has no attribute '{name}'")
            obj = getattr(self.dinov2_backbone, name)
            if isinstance(obj, nn.Parameter):
                obj.requires_grad_(False)
            elif isinstance(obj, nn.Module):
                for p in obj.parameters():
                    p.requires_grad_(False)
            else:
                raise TypeError(
                    f"freeze_dino['{name}'] is {type(obj).__name__}, "
                    "expected nn.Parameter or nn.Module"
                )

        blocks_to_freeze = set(freeze_dino_blocks) | set(lora_dino_blocks)
        for idx in blocks_to_freeze:
            for p in self.dinov2_backbone.blocks[idx].parameters():
                p.requires_grad_(False)

        if lora_dino_blocks:
            if lora_config is None or "rank" not in lora_config:
                raise ValueError(
                    "lora_dino_blocks non-empty but lora_config missing rank"
                )
            rank = int(lora_config["rank"])
            dropout = float(lora_config.get("dropout", 0.0))
            target_modules = lora_config.get("target_modules", ["attn.qkv", "attn.proj"])
            trainable_modules = lora_config.get("trainable_modules", [])

            if rank > 0:
                if "alpha" not in lora_config:
                    raise ValueError(
                        "lora_config missing alpha (required when rank > 0)"
                    )
                alpha = float(lora_config["alpha"])
                from models.lora import wrap_linear_with_lora

            for idx in lora_dino_blocks:
                blk = self.dinov2_backbone.blocks[idx]
                if rank > 0:
                    for tgt in target_modules:
                        wrap_linear_with_lora(blk, tgt, rank, alpha, dropout)
                for name in trainable_modules:
                    if not hasattr(blk, name):
                        raise ValueError(
                            f"block[{idx}] has no attribute '{name}' "
                            "(trainable_modules)"
                        )
                    obj = getattr(blk, name)
                    if isinstance(obj, nn.Parameter):
                        obj.requires_grad_(True)
                    elif isinstance(obj, nn.Module):
                        for p in obj.parameters():
                            p.requires_grad_(True)
                    else:
                        raise TypeError(
                            f"block[{idx}].{name} is {type(obj).__name__}, "
                            "expected nn.Parameter or nn.Module"
                        )

    def _dino_attn_fp32(self):
        import types

        def attn_forward_fp32(self_attn, x):
            B, N, C = x.shape
            orig_dtype = x.dtype
            qkv = self_attn.qkv(x)
            with torch.cuda.amp.autocast(enabled=False):
                qkv = qkv.float()
                q, k, v = qkv.reshape(B, N, 3, self_attn.num_heads, C // self_attn.num_heads).unbind(2)
                q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self_attn.attn_drop if self_attn.training else 0.0,
                )
                out = out.transpose(1, 2).contiguous().view(B, N, C).to(orig_dtype)
            return self_attn.proj_drop(self_attn.proj(out))

        for blk in self.dinov2_backbone.blocks:
            blk.attn.forward = types.MethodType(attn_forward_fp32, blk.attn)

    def print_trainable_parameters(self):
        encoder_modules = [self.z_proj, self.z_proj_ln, self.encoder_blocks, self.encoder_norm]
        decoder_modules = [self.decoder_embed, self.decoder_blocks, self.decoder_norm]
        groups = {
            "dinov2_backbone": self.dinov2_backbone,
            "mar_encoder": nn.ModuleList(encoder_modules),
            "dino_embed (proj)": self.dino_embed,
            "mar_decoder": nn.ModuleList(decoder_modules),
            "diffloss": self.diffloss,
            "class_embs": self.class_embs,
            "class_pos_embs": self.class_pos_embs,
            "fake_latents": self.fake_latents,
        }
        named = set()
        for g in groups.values():
            for p in g.parameters():
                named.add(id(p))

        rows = []
        total_t = total_a = 0
        for name, mod in groups.items():
            t = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            a = sum(p.numel() for p in mod.parameters())
            rows.append((name, t, a))
            total_t += t
            total_a += a

        other_t = other_a = 0
        for p in self.parameters():
            if id(p) not in named:
                other_a += p.numel()
                if p.requires_grad:
                    other_t += p.numel()
        if other_a > 0:
            rows.append(("other (pos_embed/etc)", other_t, other_a))
            total_t += other_t
            total_a += other_a

        width = max(len(r[0]) for r in rows)
        print(f"{'component':<{width}}  {'trainable':>14}  {'total':>14}  {'%':>6}")
        print("-" * (width + 42))
        for name, t, a in rows:
            pct = (100.0 * t / a) if a else 0.0
            print(f"{name:<{width}}  {t:>14,}  {a:>14,}  {pct:>5.1f}%")
        print("-" * (width + 42))
        pct_total = (100.0 * total_t / total_a) if total_a else 0.0
        print(f"{'TOTAL':<{width}}  {total_t:>14,}  {total_a:>14,}  {pct_total:>5.1f}%")

    def initialize_weights(self):
        for emb in self.class_embs:
            torch.nn.init.normal_(emb.weight, std=.02)
        for pos in self.class_pos_embs:
            torch.nn.init.normal_(pos, std=.02)
        for f in self.fake_latents:
            torch.nn.init.normal_(f, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        if self.decoder_depth > 0:
            torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        modules_to_init = [
            self.z_proj,
            self.z_proj_ln,
            self.encoder_blocks,
            self.encoder_norm,
            self.dino_embed,
            self.decoder_embed, self.decoder_blocks,
            self.decoder_norm,
        ]
        for module in modules_to_init:
            module.apply(self._init_weights)
        if not self.dinov2_pretrained:
            self.dinov2_backbone.apply(self._init_weights)

            nn.init.normal_(self.dinov2_backbone.mask_token, std=.02)
            nn.init.normal_(self.dinov2_backbone.cls_token,  std=.02)
            if self.dinov2_backbone.register_tokens is not None:
                nn.init.normal_(self.dinov2_backbone.register_tokens, std=.02)
            nn.init.trunc_normal_(self.dinov2_backbone.pos_embed, std=.02)

            for blk in self.dinov2_backbone.blocks:
                for ls in (blk.ls1, blk.ls2):
                    if hasattr(ls, 'gamma'):
                        nn.init.constant_(ls.gamma, 1e-5)

        if self.dinov2_external_pos_embed:
            with torch.no_grad():
                dpatch = self.dinov2_backbone.patch_size
                w = self.seq_w * dpatch
                h = self.seq_h * dpatch
                ref = self.dinov2_backbone.pos_embed
                dummy = torch.zeros(
                    1, 1 + self.seq_len, self.dino_embed_dim,
                    device=ref.device, dtype=ref.dtype,
                )
                pe = self.dinov2_backbone.interpolate_pos_encoding(dummy, w, h)
                self.dinov2_external_pos_embed_param.data.copy_(pe)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x

    def sample_orders(self, bsz):
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        bsz, seq_len, _ = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    # --------------------------------------------------------------------------
    # Helpers for VPT-Deep style buffer injection.
    def _make_buffer(self, i, class_embeddings, apply_z_proj_ln=False):
        """Build a buffer tensor for interval i from its class_embedding +
        pos_emb. Optionally apply z_proj_ln (encoder-region only)."""
        emb = class_embeddings[i]                               # (bsz, dim_i)
        buf = emb.unsqueeze(1).expand(-1, self.buffer_size, -1).contiguous()
        buf = buf + self.class_pos_embs[i]
        if apply_z_proj_ln:
            buf = self.z_proj_ln(buf)
        return buf

    def _maybe_end_interval(self, layer, active_interval, buffer):
        """If the currently active interval ends at `layer`, clear state."""
        if active_interval is not None and layer == self.class_emb_layers[active_interval][1]:
            return None, None
        return active_interval, buffer

    def _maybe_start_interval(self, layer, region, active_interval, buffer,
                              class_embeddings):
        """If any interval in `region` starts at `layer`, initialize a new
        buffer. Intervals are non-overlapping, so at most one matches."""
        if active_interval is not None:
            return active_interval, buffer
        for i, (s, _e) in enumerate(self.class_emb_layers):
            if s == layer and self.class_emb_regions[i] == region:
                apply_ln = (region == 'enc')
                buf = self._make_buffer(i, class_embeddings, apply_z_proj_ln=apply_ln)
                return i, buf
        return active_interval, buffer

    def _run_block(self, block, x, buffer):
        """Run a transformer block on optionally-prefixed buffer tokens."""
        if buffer is not None:
            x_in = torch.cat([buffer, x], dim=1)
        else:
            x_in = x
        if self.grad_checkpointing and self.training and not torch.jit.is_scripting():
            x_out = checkpoint(block, x_in)
        else:
            x_out = block(x_in)
        if buffer is not None:
            return x_out[:, self.buffer_size:], x_out[:, :self.buffer_size]
        return x_out, None

    def _norm_with_buffer(self, norm_mod, x, buffer):
        """Apply a LayerNorm-style module to (buffer + x) together."""
        if buffer is None:
            return norm_mod(x), None
        combined = torch.cat([buffer, x], dim=1)
        combined = norm_mod(combined)
        return combined[:, self.buffer_size:], combined[:, :self.buffer_size]

    # --------------------------------------------------------------------------
    def forward_mae_encoder(self, x, mask, class_embeddings):
        bsz = x.shape[0]
        E = self.encoder_depth

        x = self.z_proj(x)
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # Drop masked tokens: only image tokens present at this point.
        mask_bool = mask.bool()
        x = x[(~mask_bool).nonzero(as_tuple=True)].reshape(bsz, -1, self.encoder_embed_dim)

        buffer = None
        active_interval = None

        for l in range(E):
            active_interval, buffer = self._maybe_end_interval(l, active_interval, buffer)
            active_interval, buffer = self._maybe_start_interval(
                l, 'enc', active_interval, buffer, class_embeddings
            )
            x, buffer = self._run_block(self.encoder_blocks[l], x, buffer)

        # encoder_norm sees the buffer iff the interval spans through layer E-1.
        x, buffer = self._norm_with_buffer(self.encoder_norm, x, buffer)

        # Drop the buffer at the region boundary. If the interval's end is
        # beyond E, validation would have rejected it; otherwise it ends at
        # layer E (we already applied norm with buffer included).
        active_interval, buffer = self._maybe_end_interval(E, active_interval, buffer)
        assert buffer is None, (
            "Encoder buffer survived past layer E — intervals must not cross "
            "the encoder/dinov2 region boundary"
        )

        return x

    def forward_mae_decoder(self, x_enc, mask, class_embeddings):
        bsz = x_enc.shape[0]
        L = self.seq_len
        Dd = self.dino_embed_dim
        R = self.num_register
        E = self.encoder_depth
        D = self.dinov2_depth
        M = self.decoder_depth

        # 1. Project encoder output image tokens to dino dim.
        x_enc = self.dino_embed(x_enc)                          # (bsz, num_visible, Dd)

        # 2. Scatter visible tokens back into the full L-length sequence,
        #    filling masked positions with dinov2's mask_token.
        mask_bool = mask.bool()
        dino_mask_tok = self.dinov2_backbone.mask_token.to(x_enc.dtype)
        x_full = dino_mask_tok.view(1, 1, Dd).expand(bsz, L, Dd).contiguous()
        x_full[(~mask_bool).nonzero(as_tuple=True)] = x_enc.reshape(-1, Dd)

        if self.use_align_dino_embed and self.training:
            self.align_feat_pred = x_full.contiguous()          # (bsz, L, Dd)

        # 3. Dinov2-style token preparation: prepend cls, add pos_embed, then
        #    insert register tokens between cls and patches.
        cls = self.dinov2_backbone.cls_token.expand(bsz, -1, -1)
        x_img = torch.cat((cls, x_full), dim=1)                 # (bsz, 1+L, Dd)

        if self.dinov2_external_pos_embed:
            x_img = x_img + self.dinov2_external_pos_embed_param.to(x_img.dtype)
        else:
            dpatch = self.dinov2_backbone.patch_size
            w = self.seq_w * dpatch
            h = self.seq_h * dpatch
            x_img = x_img + self.dinov2_backbone.interpolate_pos_encoding(x_img, w, h)

        if self.dinov2_backbone.register_tokens is not None:
            reg = self.dinov2_backbone.register_tokens.expand(bsz, -1, -1)
            x_img = torch.cat(
                (x_img[:, :1], reg, x_img[:, 1:]), dim=1
            )                                                    # (bsz, 1+R+L, Dd)
        prefix_len = 1 + R

        # 4. Dinov2 transformer blocks with per-layer buffer transitions.
        buffer = None
        active_interval = None

        for l_dino in range(D):
            global_l = E + l_dino
            active_interval, buffer = self._maybe_end_interval(
                global_l, active_interval, buffer
            )
            active_interval, buffer = self._maybe_start_interval(
                global_l, 'dino', active_interval, buffer, class_embeddings
            )
            x_img, buffer = self._run_block(
                self.dinov2_backbone.blocks[l_dino], x_img, buffer
            )

        # Dino norm sees the buffer if the interval spans through layer E+D-1.
        x_img, buffer = self._norm_with_buffer(
            self.dinov2_backbone.norm, x_img, buffer
        )
        active_interval, buffer = self._maybe_end_interval(
            E + D, active_interval, buffer
        )
        assert buffer is None, (
            "Dino buffer survived past layer E+D — intervals must not cross "
            "the dinov2/decoder region boundary"
        )

        if self.use_repa and self.training:
            self.repa_feat_pred = x_img.contiguous()            # (bsz, 1+R+L, Dd)

        # 5. Project dino_embed_dim -> decoder_embed_dim.
        x_img = self.decoder_embed(x_img)                        # (bsz, 1+R+L, dec)

        # 6. Decoder blocks with per-layer buffer transitions.
        if M > 0:
            x_img = x_img + self.decoder_pos_embed_learned

            for l_dec in range(M):
                global_l = E + D + l_dec
                active_interval, buffer = self._maybe_end_interval(
                    global_l, active_interval, buffer
                )
                active_interval, buffer = self._maybe_start_interval(
                    global_l, 'dec', active_interval, buffer, class_embeddings
                )
                x_img, buffer = self._run_block(
                    self.decoder_blocks[l_dec], x_img, buffer
                )

        # Decoder norm sees the buffer if the interval spans through the last
        # decoder layer.
        x_img, buffer = self._norm_with_buffer(self.decoder_norm, x_img, buffer)
        active_interval, buffer = self._maybe_end_interval(
            E + D + M, active_interval, buffer
        )
        assert buffer is None, "Decoder buffer survived past the final layer"

        # 7. Strip cls + register → only image tokens fed to diffloss.
        x_img = x_img[:, prefix_len:]
        x_img = x_img + self.diffusion_pos_embed_learned
        return x_img

    def forward_loss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        if self.diffloss_fp32:
            with torch.cuda.amp.autocast(enabled=False):
                loss = self.diffloss(z=z.float(), target=target.float(), mask=mask.float())
        else:
            loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def _build_class_embeddings_train(self, labels, bsz, ref_dtype):
        """Build per-interval class embeddings with CFG label-drop applied
        uniformly across intervals (same drop mask for every interval)."""
        if self.training:
            drop_latent_mask = (torch.rand(bsz) < self.label_drop_prob)
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(ref_dtype)
        else:
            drop_latent_mask = None

        embs = []
        for i in range(len(self.class_emb_layers)):
            emb = self.class_embs[i](labels)                     # (bsz, dim_i)
            if drop_latent_mask is not None:
                emb = drop_latent_mask * self.fake_latents[i] + (1 - drop_latent_mask) * emb
            embs.append(emb)
        return embs

    def _build_class_embeddings_sample(self, labels, bsz, cfg):
        """Build per-interval class embeddings for sampling. When cfg != 1.0,
        each interval's embedding is concatenated along batch with its
        fake_latent to form the [cond, uncond] stack."""
        embs = []
        for i in range(len(self.class_emb_layers)):
            if labels is not None:
                emb = self.class_embs[i](labels)
            else:
                emb = self.fake_latents[i].repeat(bsz, 1)
            if cfg != 1.0:
                emb = torch.cat([emb, self.fake_latents[i].repeat(bsz, 1)], dim=0)
            embs.append(emb)
        return embs

    def forward(self, x, labels, imgs_pixel=None, feat=None):
        log_dict = {}

        # 1. Pre-compute REPA target features.
        repa_feat_gt = None
        if self.use_repa and self.training:
            if self.use_repa_cached_feat:
                if feat is None:
                    raise ValueError("use_repa_cached_feat=True requires `feat`")
                repa_feat_gt = feat
            else:
                dinov2_repa = self.dinov2_repa_bank[0]
                if imgs_pixel is None:
                    raise ValueError("use_repa=True requires `imgs_pixel`")
                device = imgs_pixel.device
                if self.repa_save_vram:
                    dinov2_repa.to(device)
                with torch.no_grad():
                    pix = (imgs_pixel - self.repa_mean) / self.repa_std
                    if (pix.shape[-1] != self.repa_input_size
                            or pix.shape[-2] != self.repa_input_size):
                        pix = F.interpolate(
                            pix,
                            size=(self.repa_input_size, self.repa_input_size),
                            mode="bicubic",
                            antialias=True,
                            align_corners=False,
                        )
                    t = dinov2_repa.prepare_tokens_with_masks(pix)
                    for blk in dinov2_repa.blocks:
                        t = blk(t)
                    repa_feat_gt = dinov2_repa.norm(t)
                if self.repa_save_vram:
                    dinov2_repa.cpu()

        # 2. Standard MAR forward.
        self.repa_feat_pred = None
        self.align_feat_pred = None
        bsz = x.shape[0]
        class_embeddings = self._build_class_embeddings_train(labels, bsz, ref_dtype=x.dtype)

        xp = self.patchify(x)
        gt_latents = xp.clone().detach()
        orders = self.sample_orders(bsz=xp.size(0))
        mask = self.random_masking(xp, orders)
        x_enc = self.forward_mae_encoder(xp, mask, class_embeddings)
        z = self.forward_mae_decoder(x_enc, mask, class_embeddings)
        diff_loss = self.forward_loss(z=z, target=gt_latents, mask=mask)

        log_dict["diff_loss"] = diff_loss.detach().item()
        loss = diff_loss

        # 3. Add REPA loss.
        if self.use_repa and self.training:
            pred = self.repa_feat_pred
            if pred is None:
                raise RuntimeError(
                    "use_repa=True but forward_mae_decoder did not populate "
                    "self.repa_feat_pred (training flag mismatch?)"
                )
            if pred.shape != repa_feat_gt.shape:
                raise RuntimeError(
                    f"REPA shape mismatch: pred {tuple(pred.shape)} vs "
                    f"gt {tuple(repa_feat_gt.shape)}"
                )
            if self.repa_on_unmasked:
                prefix = 1 + self.num_register
                cos_sim = F.cosine_similarity(pred, repa_feat_gt, dim=-1)
                img_weight = 1.0 - mask
                weight = torch.cat([
                    torch.ones(mask.shape[0], prefix, device=mask.device),
                    img_weight,
                ], dim=1)
                repa_loss = ((1.0 - cos_sim) * weight).sum() / weight.sum().clamp(min=1.0)
            else:
                repa_loss = (1.0 - F.cosine_similarity(pred, repa_feat_gt, dim=-1)).mean()
            loss = loss + self.repa_loss_weight * repa_loss
            log_dict["repa_loss"] = repa_loss.detach().item()
            self.repa_feat_pred = None

        # 4. Align dino_embed output on visible image tokens with patch_embed.
        if self.use_align_dino_embed and self.training:
            pred = self.align_feat_pred
            if pred is None:
                raise RuntimeError(
                    "use_align_dino_embed=True but forward_mae_decoder did not "
                    "populate self.align_feat_pred (training flag mismatch?)"
                )
            if imgs_pixel is None:
                raise ValueError("use_align_dino_embed=True requires `imgs_pixel`")

            align_input_size = self.seq_w * self.dinov2_backbone.patch_size
            with torch.no_grad():
                pix = (imgs_pixel - self.repa_mean) / self.repa_std
                if (pix.shape[-1] != align_input_size
                        or pix.shape[-2] != align_input_size):
                    pix = F.interpolate(
                        pix,
                        size=(align_input_size, align_input_size),
                        mode="bicubic",
                        antialias=True,
                        align_corners=False,
                    )
                dino_embed_gt = self.dinov2_backbone.patch_embed(pix)

            keep = 1.0 - mask
            if self.align_dino_embed_loss_type == 'cos':
                per_tok = 1.0 - F.cosine_similarity(pred, dino_embed_gt, dim=-1)
            else:
                per_tok = (pred - dino_embed_gt).pow(2).mean(dim=-1)
            align_loss = (per_tok * keep).sum() / keep.sum().clamp(min=1.0)
            loss = loss + self.align_dino_embed_loss_weight * align_loss
            log_dict["align_dino_embed_loss"] = align_loss.detach().item()
            self.align_feat_pred = None

        log_dict["loss"] = loss.detach().item()
        return loss, log_dict

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False):
        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        orders = self.sample_orders(bsz)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        for step in indices:
            cur_tokens = tokens.clone()

            class_embeddings = self._build_class_embeddings_sample(labels, bsz, cfg)
            if cfg != 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                mask = torch.cat([mask, mask], dim=0)

            x = self.forward_mae_encoder(tokens, mask, class_embeddings)
            z = self.forward_mae_decoder(x, mask, class_embeddings)

            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            z = z[mask_to_pred.nonzero(as_tuple=True)]
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError
            sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        tokens = self.unpatchify(tokens)
        return tokens

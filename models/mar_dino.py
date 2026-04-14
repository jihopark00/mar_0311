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


class MAR_DINO(nn.Module):
    """ MAR with dinov2 transformer blocks injected between the MAR encoder and decoder.
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
                 dino_attn_fp32=False,
                 diffloss_fp32=False,
                 freeze_dino=['norm', 'cls_token', 'register_tokens', 'mask_token', 'pos_embed'],
                 freeze_dino_blocks=[],
                 lora_dino_blocks=[],
                 lora_config=None,
                 use_repa=False,
                 use_repa_cached_feat=False,
                 repa_loss_weight=0.5,
                 repa_save_vram=False,
                 dinov2_repa_name=None,
                 repa_input_size=None,
                 ):
        super().__init__()

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
        # We reuse its transformer blocks + cls/register/mask/pos_embed, but NOT
        # its patch_embed — MAR feeds VAE-latent tokens in directly.
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
        self.decoder_embed_dim = decoder_embed_dim
        self.num_register = num_register

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.buffer_size = buffer_size
        self.encoder_depth = encoder_depth

        if encoder_depth > 0:
            self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
            self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # Dino embedding: projection from encoder_embed_dim to dino_embed_dim,
        # applied before the dinov2 transformer blocks.
        self.dino_embed = nn.Linear(encoder_embed_dim, dino_embed_dim, bias=True)

        # --------------------------------------------------------------------------
        # Positional embedding for buffer tokens in the dino block stage.
        # Without this, buffer tokens enter dinov2 blocks with no positional info.
        self.buffer_dino_pos_embed = nn.Parameter(torch.zeros(1, self.buffer_size, dino_embed_dim))

        # --------------------------------------------------------------------------
        # MAR decoder specifics. The dinov2 blocks run at dino_embed_dim; the MAR
        # decoder blocks run at decoder_embed_dim (independent). decoder_embed
        # projects from dino_embed_dim to decoder_embed_dim after the dinov2 blocks.
        self.decoder_embed = nn.Linear(dino_embed_dim, decoder_embed_dim, bias=True)
        self.decoder_depth = decoder_depth

        if decoder_depth > 0:
            # pos embed covers: [buffer | dino_cls | dino_registers | image_tokens]
            self.decoder_pos_embed_learned = nn.Parameter(
                torch.zeros(1, self.buffer_size + 1 + num_register + self.seq_len, decoder_embed_dim)
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

        # Apply PEFT-style tuning controls to the dinov2 backbone.
        self._apply_dino_tuning(
            freeze_dino=freeze_dino,
            freeze_dino_blocks=freeze_dino_blocks,
            lora_dino_blocks=lora_dino_blocks,
            lora_config=lora_config,
        )

        if dino_attn_fp32:
            self._dino_attn_fp32()

        # --------------------------------------------------------------------------
        # REPA: Representation Alignment with a separate frozen dinov2 on raw pixels,
        # or with pre-cached features (use_repa_cached_feat=True).
        self.use_repa = use_repa
        self.use_repa_cached_feat = use_repa_cached_feat
        self.repa_loss_weight = float(repa_loss_weight)
        self.repa_save_vram = repa_save_vram
        # populated by forward_mae_decoder when training+use_repa, cleared in forward()
        self.repa_feat_pred = None
        
        if use_repa:
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

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "use_repa", False) and not getattr(self, "use_repa_cached_feat", False):
            self.dinov2_repa_bank[0].eval()
        return self

    def _apply(self, fn, *args, **kwargs):
        # dinov2_repa_bank holds a tuple-wrapped submodule that is hidden from
        # nn.Module's parameter/state_dict tracking. Forward _apply manually so
        # model.to(device) / .cuda() / .cpu() / .float() / etc. propagate to it.
        super()._apply(fn, *args, **kwargs)
        bank = self.__dict__.get('dinov2_repa_bank', None)
        if bank is not None:
            bank[0]._apply(fn, *args, **kwargs)
        return self

    def _apply_dino_tuning(self, freeze_dino=None, freeze_dino_blocks=None,
                           lora_dino_blocks=None, lora_config=None):
        """Freeze and/or LoRA-wrap selected parts of self.dinov2_backbone.

        - freeze_dino: list of attribute names on self.dinov2_backbone whose
          parameters should be frozen. Both nn.Parameter attrs (e.g. cls_token,
          register_tokens, mask_token, pos_embed) and nn.Module attrs
          (e.g. norm, patch_embed) are accepted.
        - freeze_dino_blocks: indices into self.dinov2_backbone.blocks whose
          parameters are frozen entirely.
        - lora_dino_blocks: indices into self.dinov2_backbone.blocks; each
          listed block gets its target Linears wrapped with LoRALinear. The
          base Linear weights are frozen by LoRALinear; other params in the
          block (norm1/norm2/ls1/ls2) remain trainable.
        - lora_config keys:
            rank (int, required)
            alpha (float, required)
            target_modules (list[str], default ['attn.qkv', 'attn.proj'])
            dropout (float, default 0.0)

        Blocks not named in either list remain fully trainable.
        """
        freeze_dino = list(freeze_dino or [])
        freeze_dino_blocks = list(freeze_dino_blocks or [])
        lora_dino_blocks = list(lora_dino_blocks or [])

        # 1. Freeze named backbone attributes (parameters or submodules).
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

        # 2. A block cannot appear in both the freeze and LoRA lists.
        overlap = set(freeze_dino_blocks) & set(lora_dino_blocks)
        if overlap:
            raise ValueError(
                f"blocks {sorted(overlap)} appear in both "
                "freeze_dino_blocks and lora_dino_blocks"
            )

        # 3. Freeze whole blocks.
        for idx in freeze_dino_blocks:
            for p in self.dinov2_backbone.blocks[idx].parameters():
                p.requires_grad_(False)

        # 4. Apply LoRA to the selected blocks.
        if lora_dino_blocks:
            if not lora_config or "rank" not in lora_config or "alpha" not in lora_config:
                raise ValueError(
                    "lora_dino_blocks non-empty but lora_config missing rank/alpha"
                )
            from models.lora import wrap_linear_with_lora
            rank = int(lora_config["rank"])
            alpha = float(lora_config["alpha"])
            dropout = float(lora_config.get("dropout", 0.0))
            target_modules = lora_config.get("target_modules", ["attn.qkv", "attn.proj"])
            for idx in lora_dino_blocks:
                blk = self.dinov2_backbone.blocks[idx]
                for tgt in target_modules:
                    wrap_linear_with_lora(blk, tgt, rank, alpha, dropout)

    def _dino_attn_fp32(self):
        """Monkey-patch each DINOv2 attention block to run QKV + SDPA in fp32.

        DINOv2 ViT-g (40 blocks, LayerScale gammas up to ±10 in later blocks)
        produces NaN gradients during bf16 backward: SDPA backward is numerically
        unstable for the concentrated attention patterns these pretrained weights
        produce on OOD (VAE-latent) inputs.  Running only the attention matmul in
        fp32 fixes this with minimal extra memory (no need to keep the full
        [B, N, D] activation stream in fp32 for all 40 blocks).
        """
        import types

        def attn_forward_fp32(self_attn, x):
            B, N, C = x.shape
            orig_dtype = x.dtype
            qkv = self_attn.qkv(x)
            with torch.cuda.amp.autocast(enabled=False):
                # qkv = self_attn.qkv(x.float())
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
        """Print a per-component summary of trainable vs total parameters."""
        groups = {
            "dinov2_backbone": self.dinov2_backbone,
            "mar_encoder": nn.ModuleList(
                [self.z_proj, self.z_proj_ln, self.encoder_blocks, self.encoder_norm]
                if self.encoder_depth > 0 else
                [self.z_proj, self.encoder_blocks, self.encoder_norm]
            ),
            "dino_embed (proj)": self.dino_embed,
            "mar_decoder": nn.ModuleList([
                self.decoder_embed, self.decoder_blocks, self.decoder_norm,
            ]),
            "diffloss": self.diffloss,
            "class_emb": self.class_emb,
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

        # Catch any top-level params not covered above (e.g. encoder_pos_embed_learned,
        # decoder_pos_embed_learned, diffusion_pos_embed_learned, fake_latent).
        other_t = other_a = 0
        for p in self.parameters():
            if id(p) not in named:
                other_a += p.numel()
                if p.requires_grad:
                    other_t += p.numel()
        if other_a > 0:
            rows.append(("other (pos_embed/fake_latent)", other_t, other_a))
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
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        if self.encoder_depth > 0:
            torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.buffer_dino_pos_embed, std=.02)
        if self.decoder_depth > 0:
            torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm for modules we own.
        # Skip the dinov2 backbone so its pretrained weights are not overwritten.
        modules_to_init = [
            self.z_proj,
            self.encoder_blocks, self.encoder_norm,
            self.dino_embed,
            self.decoder_embed, self.decoder_blocks, self.decoder_norm,
            self.class_emb,
        ]
        if self.encoder_depth > 0:
            modules_to_init.append(self.z_proj_ln)
        for module in modules_to_init:
            module.apply(self._init_weights)
        if not self.dinov2_pretrained:
            self.dinov2_backbone.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
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
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, _ = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def forward_mae_encoder(self, x, mask, class_embedding):
        x = self.z_proj(x)
        bsz, _, embed_dim = x.shape

        # concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)

        if self.encoder_depth > 0:
            # encoder position embedding
            x = x + self.encoder_pos_embed_learned
            x = self.z_proj_ln(x)

        # dropping
        x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        if self.encoder_depth > 0:
            # apply Transformer blocks
            if self.grad_checkpointing and self.training and not torch.jit.is_scripting():
                for block in self.encoder_blocks:
                    x = checkpoint(block, x)
            else:
                for block in self.encoder_blocks:
                    x = block(x)
        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder(self, x, mask):
        bsz = x.shape[0]
        buf = self.buffer_size
        L = self.seq_len
        Dd = self.dino_embed_dim
        R = self.num_register

        # 1. Project encoder output to dino embedding dim
        x = self.dino_embed(x)  # (bsz, num_visible, Dd)

        # 2. Scatter visible tokens back into the full (buffer + image) sequence,
        #    filling the gaps with dinov2's mask_token.
        mask_with_buffer = torch.cat(
            [torch.zeros(bsz, buf, device=x.device), mask], dim=1
        )  # (bsz, buf + L)
        dino_mask_tok = self.dinov2_backbone.mask_token.to(x.dtype)  # (1, Dd)
        x_after_pad = dino_mask_tok.view(1, 1, Dd).expand(bsz, buf + L, Dd).contiguous()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(
            x.shape[0] * x.shape[1], Dd
        )

        # 3. Split into MAR buffer (class) tokens and image tokens.
        buffer_tokens = x_after_pad[:, :buf]          # (bsz, buf, Dd)
        image_tokens  = x_after_pad[:, buf:]          # (bsz, L, Dd)

        # 4. Dinov2-style token preparation (mirrors prepare_tokens_with_masks).
        cls = self.dinov2_backbone.cls_token.expand(bsz, -1, -1)  # (bsz, 1, Dd)
        image_tokens = torch.cat((cls, image_tokens), dim=1)      # (bsz, 1+L, Dd)

        # Apply dinov2's interpolated pos_embed. interpolate_pos_encoding expects
        # pixel extents; dinov2 recomputes the patch grid as w // patch_size.
        dpatch = self.dinov2_backbone.patch_size
        w = self.seq_w * dpatch
        h = self.seq_h * dpatch
        image_tokens = image_tokens + self.dinov2_backbone.interpolate_pos_encoding(
            image_tokens, w, h
        )

        # Insert register tokens between cls and patches (dinov2 convention).
        if self.dinov2_backbone.register_tokens is not None:
            reg = self.dinov2_backbone.register_tokens.expand(bsz, -1, -1)
            image_tokens = torch.cat(
                (image_tokens[:, :1], reg, image_tokens[:, 1:]), dim=1
            )  # (bsz, 1+R+L, Dd)

        # Add positional embedding to buffer tokens before dino blocks.
        buffer_tokens = buffer_tokens + self.buffer_dino_pos_embed

        # Prepend MAR buffer tokens to match mar.py's buffer-at-front layout.
        x = torch.cat((buffer_tokens, image_tokens), dim=1)  # (bsz, buf+1+R+L, Dd)

        # 5. Dinov2 transformer blocks (optional grad checkpointing),
        #    followed by dinov2's own final norm (matches forward_features).
        if self.grad_checkpointing and self.training and not torch.jit.is_scripting():
            for blk in self.dinov2_backbone.blocks:
                x = checkpoint(blk, x)
        else:
            for blk in self.dinov2_backbone.blocks:
                x = blk(x)
        x = self.dinov2_backbone.norm(x)

        # REPA: capture post-norm features (drop MAR buffer; keep cls+register+image).
        if self.use_repa and self.training:
            self.repa_feat_pred = x[:, buf:].contiguous()

        # 6. Project dino_embed_dim -> decoder_embed_dim, then MAR decoder blocks.
        x = self.decoder_embed(x)  # (bsz, buf+1+R+L, decoder_embed_dim)

        if self.decoder_depth > 0:
            x = x + self.decoder_pos_embed_learned

            if self.grad_checkpointing and self.training and not torch.jit.is_scripting():
                for blk in self.decoder_blocks:
                    x = checkpoint(blk, x)
            else:
                for blk in self.decoder_blocks:
                    x = blk(x)
        x = self.decoder_norm(x)

        # 7. Strip buffer + cls + register → only image tokens fed to diffloss.
        x = x[:, buf + 1 + R:]
        x = x + self.diffusion_pos_embed_learned
        return x

    def forward_loss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        if self.diffloss_fp32:
            # Run DiffLoss in fp32: IDDPM's normal_kl / discretized_gaussian_log_likelihood
            # overflow in bf16 (exp of learned logvar), producing NaN.
            with torch.cuda.amp.autocast(enabled=False):
                loss = self.diffloss(z=z.float(), target=target.float(), mask=mask.float())
        else:
            loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

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
                    repa_feat_gt = dinov2_repa.norm(t)  # (bsz, 1+R+L, D)
                if self.repa_save_vram:
                    dinov2_repa.cpu()

        # 2. Standard MAR forward (forward_mae_decoder may populate repa_feat_pred).
        self.repa_feat_pred = None  # safety: clear stale state
        class_embedding = self.class_emb(labels)
        xp = self.patchify(x)
        gt_latents = xp.clone().detach()
        orders = self.sample_orders(bsz=xp.size(0))
        mask = self.random_masking(xp, orders)
        x_enc = self.forward_mae_encoder(xp, mask, class_embedding)
        z = self.forward_mae_decoder(x_enc, mask)
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
            repa_loss = (1.0 - F.cosine_similarity(pred, repa_feat_gt, dim=-1)).mean()
            loss = loss + self.repa_loss_weight * repa_loss
            log_dict["repa_loss"] = repa_loss.detach().item()
            self.repa_feat_pred = None

        log_dict["loss"] = loss.detach().item()
        return loss, log_dict

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False):

        # init and sample generation orders
        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        orders = self.sample_orders(bsz)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()

            # class embedding and CFG
            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)
            if not cfg == 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask = torch.cat([mask, mask], dim=0)

            # mae encoder
            x = self.forward_mae_encoder(tokens, mask, class_embedding)

            # mae decoder
            z = self.forward_mae_decoder(x, mask)

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError
            sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        # unpatchify
        tokens = self.unpatchify(tokens)
        return tokens

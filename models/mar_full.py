"""
MAR with full-token denoising.

Instead of denoising each token independently, this variant denoises all tokens
together using a full-sequence transformer. Previously generated tokens can
attend to and influence the denoising of new tokens.
"""

from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import Block

from models.flowloss_full import FlowLossFull


def mask_by_order(mask_len, order, bsz, seq_len):
    """Create mask based on generation order."""
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(
        masking, dim=-1, index=order[:, :mask_len.long()],
        src=torch.ones(bsz, seq_len).cuda()
    ).bool()
    return masking


class MARFull(nn.Module):
    """
    MAR with full-token denoising.

    Key differences from MAR:
    - Uses FlowLossFull which processes full sequences [B, L, C]
    - Does NOT flatten tokens to [B*L, C] for loss computation
    - During sampling, previously generated tokens are visible to the denoiser
    """

    def __init__(
        self,
        img_size=256,
        vae_stride=16,
        patch_size=1,
        encoder_embed_dim=1024,
        encoder_depth=16,
        encoder_num_heads=16,
        decoder_embed_dim=1024,
        decoder_depth=16,
        decoder_num_heads=16,
        mlp_ratio=4.,
        norm_layer=nn.LayerNorm,
        vae_embed_dim=16,
        mask_ratio_min=0.7,
        label_drop_prob=0.1,
        class_num=1000,
        attn_dropout=0.1,
        proj_dropout=0.1,
        buffer_size=64,
        diffusion_batch_mul=4,
        grad_checkpointing=False,
        # DiffLoss (FlowLossFull) specific
        diffloss_kwargs=None,
    ):
        super().__init__()

        # Default diffloss kwargs
        if diffloss_kwargs is None:
            diffloss_kwargs = {
                "num_sampling_steps": 50,
                "net_kwargs": {
                    "d_model": 512,
                    "depth": 6,
                    "num_heads": 8,
                    "cond_method": "adaln",
                },
            }

        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim
        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size ** 2
        self.grad_checkpointing = grad_checkpointing

        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # Masking ratio generator
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25
        )

        # MAR encoder
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim)
        )
        self.encoder_blocks = nn.ModuleList([
            Block(
                encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True,
                norm_layer=norm_layer, proj_drop=proj_dropout, attn_drop=attn_dropout
            )
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # MAR decoder
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim)
        )
        self.decoder_blocks = nn.ModuleList([
            Block(
                decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,
                norm_layer=norm_layer, proj_drop=proj_dropout, attn_drop=attn_dropout
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len, decoder_embed_dim)
        )

        self.initialize_weights()

        # Full-sequence Diff Loss
        _diffloss_kwargs = dict(diffloss_kwargs)
        _diffloss_kwargs["target_channels"] = self.token_embed_dim
        _diffloss_kwargs["z_channels"] = decoder_embed_dim
        # Pass seq_len to the network
        if "net_kwargs" not in _diffloss_kwargs:
            _diffloss_kwargs["net_kwargs"] = {}
        _diffloss_kwargs["net_kwargs"]["seq_len"] = self.seq_len
        self.diffloss = FlowLossFull(**_diffloss_kwargs)
        self.diffusion_batch_mul = diffusion_batch_mul

    def initialize_weights(self):
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        """Convert image to patches: [B, C, H, W] -> [B, L, D]"""
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p
        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x

    def unpatchify(self, x):
        """Convert patches to image: [B, L, D] -> [B, C, H, W]"""
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w
        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x

    def sample_orders(self, bsz):
        """Generate batch of random generation orders."""
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        """Generate token mask with truncated Gaussian ratio."""
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(
            mask, dim=-1, index=orders[:, :num_masked_tokens],
            src=torch.ones(bsz, seq_len, device=x.device)
        )
        return mask

    def forward_mae_encoder(self, x, mask, class_embedding):
        """MAR encoder: process visible tokens."""
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape

        # Concat buffer
        x = torch.cat([
            torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x
        ], dim=1)
        mask_with_buffer = torch.cat([
            torch.zeros(x.size(0), self.buffer_size, device=x.device), mask
        ], dim=1)

        # Random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = (
                drop_latent_mask * self.fake_latent +
                (1 - drop_latent_mask) * class_embedding
            )

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)

        # Position embedding
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # Drop masked tokens
        x = x[(1 - mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        # Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.encoder_blocks:
                x = block(x)
        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder(self, x, mask):
        """MAR decoder: reconstruct all positions."""
        x = self.decoder_embed(x)
        mask_with_buffer = torch.cat([
            torch.zeros(x.size(0), self.buffer_size, device=x.device), mask
        ], dim=1)

        # Pad with mask tokens
        mask_tokens = self.mask_token.repeat(
            mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1
        ).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(
            x.shape[0] * x.shape[1], x.shape[2]
        )

        # Position embedding
        x = x_after_pad + self.decoder_pos_embed_learned

        # Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.decoder_blocks:
                x = block(x)
        x = self.decoder_norm(x)

        # Remove buffer and add diffusion pos embed
        x = x[:, self.buffer_size:]
        x = x + self.diffusion_pos_embed_learned
        return x

    def forward_loss(self, z, target, mask):
        """
        Compute flow loss on full sequences.

        Key difference from MAR: NO flattening to [B*L, C]!
        Passes full sequences directly to FlowLossFull.

        Args:
            z: [B, L, D] - conditioning from decoder
            target: [B, L, C] - ground truth tokens
            mask: [B, L] - binary mask (1 = compute loss)
        """
        # Repeat for diffusion_batch_mul (variance reduction)
        if self.diffusion_batch_mul > 1:
            target = target.repeat(self.diffusion_batch_mul, 1, 1)  # [B*mul, L, C]
            z = z.repeat(self.diffusion_batch_mul, 1, 1)  # [B*mul, L, D]
            mask = mask.repeat(self.diffusion_batch_mul, 1)  # [B*mul, L]

        # Pass full sequences (not flattened!)
        loss = self.diffloss(target=target, z=z, mask=mask)
        return loss

    def forward(self, imgs, labels):
        """Training forward pass."""
        # Class embedding
        class_embedding = self.class_emb(labels)

        # Patchify and mask tokens
        x = self.patchify(imgs)
        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz=x.size(0))
        mask = self.random_masking(x, orders)

        # MAR encoder
        x = self.forward_mae_encoder(x, mask, class_embedding)

        # MAR decoder
        z = self.forward_mae_decoder(x, mask)

        # Flow loss (full sequence)
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

    def sample_tokens(
        self,
        bsz,
        num_iter=64,
        cfg=1.0,
        cfg_schedule="linear",
        labels=None,
        temperature=1.0,
        progress=False
    ):
        """
        Generate tokens with full-sequence denoising.

        Key insight: Previously generated tokens are VISIBLE during denoising
        of new tokens. They are passed to the diffloss.sample() method as x_known.
        """
        grad_checkpointing = self.grad_checkpointing
        diffloss_grad_checkpointing = self.diffloss.net.grad_checkpointing

        self.grad_checkpointing = False
        self.diffloss.net.grad_checkpointing = False

        # Initialize
        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        orders = self.sample_orders(bsz)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)

        for step in indices:
            cur_tokens = tokens.clone()

            # Class embedding and CFG
            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)

            if cfg != 1.0:
                tokens_input = torch.cat([tokens, tokens], dim=0)
                class_embedding_input = torch.cat([
                    class_embedding, self.fake_latent.repeat(bsz, 1)
                ], dim=0)
                mask_input = torch.cat([mask, mask], dim=0)
            else:
                tokens_input = tokens
                class_embedding_input = class_embedding
                mask_input = mask

            # MAR encoder
            x = self.forward_mae_encoder(tokens_input, mask_input, class_embedding_input)

            # MAR decoder
            z = self.forward_mae_decoder(x, mask_input)  # [B or 2B, L, D]

            # Mask ratio for next round
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()
            mask_len = torch.maximum(
                torch.Tensor([1]).cuda(),
                torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len)
            )

            # Get masking for next iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)

            if step >= num_iter - 1:
                mask_to_pred = mask.bool()
            else:
                mask_to_pred = torch.logical_xor(mask.bool(), mask_next.bool())

            # CFG schedule
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError

            # Sample ALL tokens together with full-sequence denoising
            # Previously generated tokens (cur_tokens at unmasked positions) serve as context!
            if cfg != 1.0:
                x_known = torch.cat([cur_tokens, cur_tokens], dim=0)
            else:
                x_known = cur_tokens

            sampled_tokens = self.diffloss.sample(
                z=z,
                x_known=x_known,
                mask=mask_input,  # Which positions to denoise (1 = denoise)
                temperature=temperature,
                cfg=cfg_iter if cfg != 1.0 else 1.0,
            )

            # Update only the newly predicted positions
            cur_tokens[mask_to_pred] = sampled_tokens[mask_to_pred]
            tokens = cur_tokens.clone()
            mask = mask_next.float()

        # Unpatchify
        tokens = self.unpatchify(tokens)

        self.grad_checkpointing = grad_checkpointing
        self.diffloss.net.grad_checkpointing = diffloss_grad_checkpointing

        return tokens


def mar_full_base(**kwargs):
    model = MARFull(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def mar_full_large(**kwargs):
    model = MARFull(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def mar_full_huge(**kwargs):
    model = MARFull(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model

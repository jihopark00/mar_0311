"""
Full-token denoising for MAR.

Instead of denoising each token independently, this module denoises all tokens
together with cross-token attention, allowing the model to capture dependencies
between tokens during the denoising process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math

from util.model_util import get_2d_sincos_pos_embed, RMSNorm


def modulate_seq(x, shift, scale):
    """Modulate for sequence input: x is [B, L, D], shift/scale are [B, D]"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def modulate_seq_per_token(x, shift, scale):
    """Modulate per-token: x is [B, L, D], shift/scale are [B, L, D]"""
    return x * (1 + scale) + shift


def modulate(x, shift, scale):
    """Modulate: x is [B, D], shift/scale are [B, D]"""
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t * 1000.0
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, dim: int, hidden_dim: int, drop=0.0, bias=True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class Attention(nn.Module):
    """Multi-head self-attention with optional attention mask and KV caching."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True,
                 attn_drop=0., proj_drop=0., use_fused_attn=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_fused_attn = use_fused_attn

        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None, kv_cache=None, cache_valid_mask=None):
        """
        Args:
            x: [B, N, C] input tokens
            attn_mask: [B, N, N] or [B, H, N, N] boolean mask
                       True = CAN attend, False = BLOCKED
            kv_cache: Optional tuple (cached_k, cached_v), each [B, H, N, head_dim]
            cache_valid_mask: [B, N] boolean mask, True = use cached K,V for this position
                              (typically True for known/unmasked positions)

        Returns:
            output: [B, N, C]
            new_kv_cache: tuple (k, v) for caching, or None if no caching
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, H, N, head_dim]

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply KV cache: use cached values for positions where cache_valid_mask is True
        if kv_cache is not None and cache_valid_mask is not None:
            cached_k, cached_v = kv_cache
            # cache_valid_mask: [B, N] -> [B, 1, N, 1] for broadcasting
            mask_expand = cache_valid_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
            k = torch.where(mask_expand, cached_k, k)
            v = torch.where(mask_expand, cached_v, v)

        if self.use_fused_attn:
            # F.scaled_dot_product_attention expects [B, H, N, N] or [B, 1, N, N]
            # Expand [B, N, N] -> [B, 1, N, N] for broadcasting
            if attn_mask is not None and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # [B, 1, N, N]
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                # Convert boolean mask to float: False -> -inf, True -> 0
                if attn_mask.dtype == torch.bool:
                    attn_mask_expanded = attn_mask.unsqueeze(1) if attn_mask.dim() == 3 else attn_mask
                    attn_mask_expanded = torch.where(attn_mask_expanded, 0.0, float('-inf'))
                else:
                    attn_mask_expanded = attn_mask
                attn = attn + attn_mask_expanded
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        # Return output and new cache (k, v after potential cache merge)
        new_kv_cache = (k, v)
        return out, new_kv_cache


class FullTransformerBlock(nn.Module):
    """
    Transformer block with full self-attention and per-token AdaLN conditioning.

    Args:
        d_model: Hidden dimension
        num_heads: Number of attention heads
        d_cond: Conditioning dimension
        mlp_ratio: MLP hidden dim multiplier
        cond_method: "adaln" or "residual"
    """

    def __init__(
        self,
        d_model,
        num_heads,
        d_cond,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        use_fused_attn=True,
        cond_method="adaln",
    ):
        super().__init__()
        self.cond_method = cond_method
        self.d_model = d_model

        self.norm1 = RMSNorm(d_model, eps=1e-6)
        self.attn = Attention(
            d_model, num_heads=num_heads, qkv_bias=True, qk_norm=True,
            attn_drop=attn_drop, proj_drop=proj_drop, use_fused_attn=use_fused_attn
        )
        self.norm2 = RMSNorm(d_model, eps=1e-6)
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = SwiGLUFFN(d_model, mlp_hidden_dim, drop=proj_drop)

        # AdaLN modulation: 6 parameters per token (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        # Input: [B, L, d_cond] -> Output: [B, L, 6 * d_model]
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_cond, 6 * d_model, bias=True)
        )

    def forward(self, x, cond, attn_mask=None, kv_cache=None, cache_valid_mask=None):
        """
        Args:
            x: [B, L, d_model] - input tokens
            cond: [B, L, d_cond] - per-token conditioning (time + condition)
            attn_mask: Optional attention mask
            kv_cache: Optional tuple (cached_k, cached_v) from previous ODE step
            cache_valid_mask: [B, L] boolean, True = use cached K,V

        Returns:
            x: [B, L, d_model]
            new_kv_cache: tuple (k, v) for next step
        """
        # Get AdaLN parameters from per-token conditioning: [B, L, d_cond] -> [B, L, 6*d_model]
        adaln_out = self.adaLN_modulation(cond)  # [B, L, 6*d_model]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            adaln_out.chunk(6, dim=-1)  # Each: [B, L, d_model]

        # Self-attention with per-token AdaLN and KV caching
        attn_input = modulate_seq_per_token(self.norm1(x), shift_msa, scale_msa)
        attn_out, new_kv_cache = self.attn(
            attn_input,
            attn_mask=attn_mask,
            kv_cache=kv_cache,
            cache_valid_mask=cache_valid_mask
        )
        x = x + gate_msa * attn_out

        # MLP with per-token AdaLN
        x = x + gate_mlp * self.mlp(
            modulate_seq_per_token(self.norm2(x), shift_mlp, scale_mlp)
        )

        return x, new_kv_cache


class FullTransformerFinalLayer(nn.Module):
    """Final layer with per-token AdaLN for FullTransformer."""

    def __init__(self, d_model, out_dim, d_cond):
        super().__init__()
        self.norm_final = RMSNorm(d_model, eps=1e-6)
        self.linear = nn.Linear(d_model, out_dim, bias=True)
        # Per-token modulation: [B, L, d_cond] -> [B, L, 2*d_model]
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_cond, 2 * d_model, bias=True)
        )

    def forward(self, x, cond):
        """
        Args:
            x: [B, L, d_model]
            cond: [B, L, d_cond] - per-token conditioning
        """
        adaln_out = self.adaLN_modulation(cond)  # [B, L, 2*d_model]
        shift, scale = adaln_out.chunk(2, dim=-1)  # Each: [B, L, d_model]
        x = modulate_seq_per_token(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class FullTransformer(nn.Module):
    """
    Full-sequence Transformer for denoising all tokens together.

    Unlike per-token denoisers, this allows cross-token communication
    during denoising via self-attention across the full sequence.

    Args:
        in_channels: Token embedding dimension
        z_channels: Conditioning dimension from MAR decoder
        d_model: Transformer hidden dimension
        d_cond: Conditioning embedding dimension (defaults to d_model)
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        seq_len: Sequence length (required for reshaping flattened inputs)
        cond_method: "adaln" (per-token AdaLN) or "residual" (per-token add)
    """

    def __init__(
        self,
        in_channels,
        z_channels,
        d_model=512,
        d_cond=None,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        grad_checkpointing=False,
        use_fused_attn=True,
        seq_len=256,
        cond_method="adaln",
        cond_residual_layer = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.d_cond = d_cond if d_cond is not None else d_model
        if cond_method == "residual":
            assert self.d_cond == d_model, f"residual mode requires d_cond == d_model, got d_cond={self.d_cond}, d_model={d_model}"
        self.depth = depth
        self.seq_len = seq_len
        self.grad_checkpointing = grad_checkpointing
        self.cond_method = cond_method
        self.cond_residual_layer = cond_residual_layer

        # Input projection: token dim -> d_model
        self.x_embed = nn.Linear(in_channels, d_model)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))

        # Time embedding: [B] -> [B, d_cond]
        self.time_embed = TimestepEmbedder(self.d_cond)

        # Per-token condition embedding: [B, L, z_channels] -> [B, L, d_cond]
        self.cond_embed = nn.Linear(z_channels, self.d_cond)

        # # Per-token residual conditioning projection (for residual mode)
        # if cond_method == "residual":
        #     self.cond_residual_proj = nn.Linear(z_channels, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            FullTransformerBlock(
                d_model, num_heads, self.d_cond, mlp_ratio,
                attn_drop, proj_drop, use_fused_attn, cond_method
            )
            for _ in range(depth)
        ])

        # Final layer
        self.final_layer = FullTransformerFinalLayer(d_model, in_channels, self.d_cond)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize positional embedding with 2D sin-cos
        seq_h = int(math.sqrt(self.seq_len))
        if seq_h * seq_h == self.seq_len:
            pos_embed = get_2d_sincos_pos_embed(self.d_model, seq_h)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        else:
            nn.init.normal_(self.pos_embed, std=0.02)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c, attn_mask=None, kv_cache=None, cache_valid_mask=None):
        """
        Forward pass for full-sequence denoising.

        Args:
            x: [B*L, C] or [B, L, C] - noisy tokens
            t: [B] or [B, L] - timesteps (per batch or per token)
            c: [B*L, D] or [B, L, D] - conditioning from MAR decoder
            attn_mask: Optional attention mask
            kv_cache: Optional list of (k, v) tuples, one per layer
            cache_valid_mask: [B, L] boolean, True = use cached K,V for this position

        Returns:
            output: [B*L, C] or [B, L, C] - predicted clean tokens (matches input shape)
            new_kv_cache: list of (k, v) tuples for each layer, or None if not caching
        """
        # Handle both flattened and sequence inputs
        input_is_flat = (x.dim() == 2)

        if input_is_flat:
            N, C = x.shape
            L = self.seq_len
            B = N // L
            x = x.reshape(B, L, C)
            c = c.reshape(B, L, -1)
        else:
            B, L, C = x.shape

        # Embed input tokens
        x = self.x_embed(x)  # [B, L, d_model]
        x = x + self.pos_embed

        # Time embedding: handle both [B] and [B, L] timesteps
        if t.dim() == 1:
            # [B] -> [B, d_cond] -> broadcast to [B, L, d_cond]
            t_emb = self.time_embed(t)  # [B, d_cond]
            t_emb = t_emb.unsqueeze(1).expand(-1, L, -1)  # [B, L, d_cond]
        else:
            # [B, L] -> flatten -> embed -> reshape to [B, L, d_cond]
            t_flat = t.reshape(-1)  # [B*L]
            t_emb = self.time_embed(t_flat)  # [B*L, d_cond]
            t_emb = t_emb.reshape(B, L, -1)  # [B, L, d_cond]

        # Condition embedding
        if self.cond_method == "adaln":
            # Per-token condition embedding: [B, L, D] -> [B, L, d_cond]
            c_emb = self.cond_embed(c)  # [B, L, d_cond]
            cond = t_emb + c_emb  # Combined per-token conditioning: [B, L, d_cond]
        elif self.cond_method == "residual":  # residual
            # Add per-token condition directly to tokens (d_cond == d_model enforced)
            c_emb = self.cond_embed(c)  # [B, L, d_cond == d_model]
            # x = x + c_emb
            # Use only time for AdaLN (still per-token broadcast)
            cond = t_emb  # [B, L, d_cond]

        # Transformer blocks with KV caching
        new_kv_cache = []
        use_cache = (kv_cache is not None) or (cache_valid_mask is not None)

        if self.grad_checkpointing and self.training and not torch.jit.is_scripting():
            # No caching during training with checkpointing
            for i, block in enumerate(self.blocks):
                if self.cond_method == 'residual' and self.cond_residual_layer == i:
                    x = x + c_emb
                x, _ = checkpoint(
                    block, x, cond, attn_mask, None, None,
                    use_reentrant=False
                )
        else:
            for i, block in enumerate(self.blocks):
                if self.cond_method == 'residual' and self.cond_residual_layer == i:
                    x = x + c_emb
                    
                layer_cache = kv_cache[i] if kv_cache is not None else None
                x, layer_new_cache = block(
                    x, cond, attn_mask,
                    kv_cache=layer_cache,
                    cache_valid_mask=cache_valid_mask
                )
                if use_cache:
                    new_kv_cache.append(layer_new_cache)

        # Final layer
        x = self.final_layer(x, cond)  # [B, L, C]

        # Restore original shape
        if input_is_flat:
            x = x.reshape(N, C)

        return x, (new_kv_cache if use_cache else None)


def build_attn_mask_for_known_tokens(mask, num_heads=None):
    """
    Build attention mask where known tokens do NOT attend to masked tokens.

    This enables KV caching: known tokens' representations become stable
    (independent of masked tokens being denoised).

    Args:
        mask: [B, L] where 1=masked (unknown, to denoise), 0=known (visible)
        num_heads: if provided, expand to [B, num_heads, L, L]

    Returns:
        attn_mask: [B, L, L] or [B, num_heads, L, L] boolean mask
                   True = CAN attend, False = BLOCKED

    Attention pattern:
                  Key
                  Known  Masked
    Query  Known   ✓      ✗     (known queries ignore masked keys)
           Masked  ✓      ✓     (masked queries see everything)
    """
    # mask: [B, L], 1=masked, 0=known
    is_known_query = (mask == 0).unsqueeze(-1)  # [B, L, 1]
    is_masked_key = (mask == 1).unsqueeze(-2)   # [B, 1, L]

    # Block attention when: query is known AND key is masked
    block_attention = is_known_query & is_masked_key  # [B, L, L]
    attn_mask = ~block_attention  # [B, L, L], True = can attend

    if num_heads is not None:
        attn_mask = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)  # [B, H, L, L]

    return attn_mask


class FlowLossFull(nn.Module):
    """
    Flow Matching Loss for full-sequence denoising.

    Key differences from FlowLoss:
    - Operates on sequences [B, L, C] instead of individual tokens [N, C]
    - Single timestep t per batch item, broadcast across all tokens
    - Loss computed only on masked positions
    - Attention mask: known tokens don't attend to masked tokens (enables KV caching)

    Args:
        target_channels: Token embedding dimension
        z_channels: Conditioning dimension from MAR decoder
        num_sampling_steps: Number of ODE integration steps for sampling
        net_class: Network class name (should be "FullTransformer")
        net_kwargs: Additional kwargs for the network
    """

    def __init__(
        self,
        target_channels,
        z_channels,
        num_sampling_steps=50,
        net_class="FullTransformer",
        net_kwargs=None,
        P_mean=0.0,
        P_std=1.0,
        t_eps=0.02,
        noise_scale=1.0,
        sampling_method="euler",
        time_shift_scale=1.0,
    ):
        super().__init__()

        self.in_channels = target_channels

        # Build network
        _net_kwargs = dict(net_kwargs) if net_kwargs is not None else {}
        _net_kwargs["in_channels"] = target_channels
        _net_kwargs["z_channels"] = z_channels

        # Get network class
        if net_class == "FullTransformer":
            self.net = FullTransformer(**_net_kwargs)
        else:
            self.net = eval(net_class)(**_net_kwargs)

        # Flow matching params
        self.P_mean = P_mean
        self.P_std = P_std
        self.t_eps = t_eps
        self.noise_scale = noise_scale

        # Sampling params
        self.time_shift_scale = time_shift_scale
        self.num_sampling_steps = int(num_sampling_steps)
        self.sampling_method = sampling_method

    def sample_t(self, n: int, device=None):
        """Sample timesteps from logit-normal distribution."""
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, target, z, mask=None):
        """
        Compute flow matching loss on full sequences.

        Args:
            target: [B, L, C] - target token latents
            z: [B, L, D] - conditioning from MAR decoder
            mask: [B, L] - binary mask (1 = masked/unknown, 0 = known)
                          Only masked positions get noise, known positions stay clean.

        Returns:
            Scalar loss value
        """
        B, L, C = target.shape

        # Sample ONE timestep per batch item (not per token!)
        t = self.sample_t(B, device=target.device)  # [B]
        t_expand = t.view(B, 1, 1)  # [B, 1, 1] for broadcasting

        # Sample noise
        e = torch.randn_like(target) * self.noise_scale  # [B, L, C]

        # Flow matching interpolation for noisy tokens
        z_t_noisy = t_expand * target + (1 - t_expand) * e  # [B, L, C]

        # Per-token timestep: t for masked, 1.0 for known
        # Build attention mask: known tokens don't attend to masked tokens
        if mask is not None:
            mask_expand = mask.unsqueeze(-1)  # [B, L, 1]
            # mask=1 (unknown) -> use sampled t, mask=0 (known) -> use t=1.0
            t_per_token = torch.where(mask_expand.bool(), t_expand, torch.ones_like(t_expand))  # [B, L, 1]
            # mask=1 (unknown) -> noisy, mask=0 (known) -> clean target
            z_t = torch.where(mask_expand.bool(), z_t_noisy, target)
            # Attention mask: known queries don't attend to masked keys
            attn_mask = build_attn_mask_for_known_tokens(mask)  # [B, L, L]
        else:
            t_per_token = t_expand.expand(B, L, 1)  # [B, L, 1]
            z_t = z_t_noisy
            attn_mask = None

        # Target velocity: v = (target - z_t) / (1 - t)
        # For known positions: v = 0 (since z_t = target and t=1.0)
        v = (target - z_t) / (1 - t_per_token).clamp_min(self.t_eps)  # [B, L, C]

        # Network prediction (x-prediction) with per-token timesteps
        # No KV caching during training
        x_pred, _ = self.net(z_t, t_per_token.squeeze(-1), z, attn_mask=attn_mask)  # [B, L, C]

        # Predicted velocity from x-prediction
        v_pred = (x_pred - z_t) / (1 - t_per_token).clamp_min(self.t_eps)  # [B, L, C]

        # Velocity loss
        loss = (v - v_pred) ** 2  # [B, L, C]
        loss = loss.mean(dim=-1)  # [B, L]

        # Apply mask: only compute loss on masked (unknown) positions
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            loss = loss.mean()

        return loss

    @torch.no_grad()
    def sample(self, z, x_known=None, mask=None, temperature=1.0, cfg=1.0, use_kv_cache=False):
        """
        Generate samples using ODE integration with full-sequence denoising.

        Args:
            z: [B, L, D] - conditioning from MAR decoder
                When cfg != 1.0, z is already doubled as [z_cond, z_uncond]
            x_known: [B, L, C] - previously generated tokens (for unmasked positions)
            mask: [B, L] - positions to denoise (1 = denoise, 0 = keep known)
            temperature: Noise temperature
            cfg: Classifier-free guidance scale
            use_kv_cache: Whether to use KV caching for known tokens

        Returns:
            [B, L, C] - sampled tokens
        """
        device = z.device

        if cfg != 1.0:
            # z is [z_cond, z_uncond], handle CFG
            half_B = z.shape[0] // 2
            B, L = half_B, z.shape[1]

            # Initialize with same noise for both halves
            noise = temperature * self.noise_scale * torch.randn(B, L, self.in_channels, device=device)
            x = torch.cat([noise, noise], dim=0)  # [2B, L, C]

            full_B = z.shape[0]
        else:
            B, L = z.shape[0], z.shape[1]
            x = temperature * self.noise_scale * torch.randn(B, L, self.in_channels, device=device)
            full_B = B

        # Replace unmasked positions with known tokens
        if x_known is not None and mask is not None:
            mask_expand = mask.unsqueeze(-1).bool()  # [B, L, 1]
            x = torch.where(mask_expand, x, x_known)

        # Build attention mask: known tokens don't attend to masked tokens
        # This matches training behavior
        if mask is not None:
            attn_mask = build_attn_mask_for_known_tokens(mask)  # [B, L, L]
            # cache_valid_mask: True for known positions (mask == 0)
            cache_valid_mask = (mask == 0) if use_kv_cache else None  # [B, L]
        else:
            attn_mask = None
            cache_valid_mask = None

        # Timesteps from 0 to 1
        timesteps = torch.linspace(0.0, 1.0, self.num_sampling_steps + 1, device=device)
        timesteps = timesteps / (self.time_shift_scale - (self.time_shift_scale - 1) * timesteps)

        if self.sampling_method == "euler":
            stepper = self._euler_step
        elif self.sampling_method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError(f"Unknown sampling method: {self.sampling_method}")

        # KV cache: populated on first step, reused on subsequent steps
        kv_cache = None

        # ODE integration
        for i in range(self.num_sampling_steps - 1):
            t = timesteps[i].expand(full_B)
            t_next = timesteps[i + 1].expand(full_B)
            x, kv_cache = stepper(
                x, t, t_next, z, cfg,
                attn_mask=attn_mask,
                kv_cache=kv_cache,
                cache_valid_mask=cache_valid_mask
            )

            # Keep known tokens fixed at each step
            if x_known is not None and mask is not None:
                x = torch.where(mask_expand, x, x_known)

        # Last step with euler
        t = timesteps[-2].expand(full_B)
        t_next = timesteps[-1].expand(full_B)
        x, _ = self._euler_step(
            x, t, t_next, z, cfg,
            attn_mask=attn_mask,
            kv_cache=kv_cache,
            cache_valid_mask=cache_valid_mask
        )

        # Final keep known tokens
        if x_known is not None and mask is not None:
            x = torch.where(mask_expand, x, x_known)

        # Remove CFG duplicate if needed
        if cfg != 1.0:
            x, _ = x.chunk(2, dim=0)

        return x

    @torch.no_grad()
    def _forward_sample(self, x, t, z, cfg=1.0, attn_mask=None, kv_cache=None, cache_valid_mask=None):
        """Forward pass for sampling with optional CFG and KV caching."""
        t_input = t.view(-1, 1, 1)  # [B, 1, 1]

        # x-prediction with KV caching
        x_pred, new_kv_cache = self.net(
            x, t, z,
            attn_mask=attn_mask,
            kv_cache=kv_cache,
            cache_valid_mask=cache_valid_mask
        )
        v_pred = (x_pred - x) / (1.0 - t_input).clamp_min(self.t_eps)

        if cfg != 1.0:
            # Split cond/uncond predictions
            v_cond, v_uncond = v_pred.chunk(2, dim=0)
            # CFG interpolation
            v_cfg = v_uncond + cfg * (v_cond - v_uncond)
            # Duplicate back for both halves
            v_pred = torch.cat([v_cfg, v_cfg], dim=0)

        return v_pred, new_kv_cache

    @torch.no_grad()
    def _euler_step(self, x, t, t_next, z, cfg=1.0, attn_mask=None, kv_cache=None, cache_valid_mask=None):
        """Euler integration step with KV caching."""
        v_pred, new_kv_cache = self._forward_sample(
            x, t, z, cfg,
            attn_mask=attn_mask,
            kv_cache=kv_cache,
            cache_valid_mask=cache_valid_mask
        )
        t_input = t.view(-1, 1, 1)
        t_next_input = t_next.view(-1, 1, 1)
        x_next = x + (t_next_input - t_input) * v_pred
        return x_next, new_kv_cache

    @torch.no_grad()
    def _heun_step(self, x, t, t_next, z, cfg=1.0, attn_mask=None, kv_cache=None, cache_valid_mask=None):
        """Heun integration step (2nd order) with KV caching."""
        t_input = t.view(-1, 1, 1)
        t_next_input = t_next.view(-1, 1, 1)

        # First euler step - this builds/uses the cache
        v_pred_t, new_kv_cache = self._forward_sample(
            x, t, z, cfg,
            attn_mask=attn_mask,
            kv_cache=kv_cache,
            cache_valid_mask=cache_valid_mask
        )
        x_next_euler = x + (t_next_input - t_input) * v_pred_t

        # Evaluate at next point - reuse the same cache (known tokens haven't changed)
        v_pred_t_next, _ = self._forward_sample(
            x_next_euler, t_next, z, cfg,
            attn_mask=attn_mask,
            kv_cache=new_kv_cache,
            cache_valid_mask=cache_valid_mask
        )

        # Average velocities
        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        x_next = x + (t_next_input - t_input) * v_pred
        return x_next, new_kv_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
import numpy as np

from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm


class FlowLoss(nn.Module):
    """Flow Matching Loss with x-prediction (JIT formulation)"""
    def __init__(self,
        target_channels, num_sampling_steps,
        z_channels=None,  # Accept z_channels for MAR interface compatibility
        net_class = 'SimplePiT',
        net_kwargs= {},
        P_mean=0.0, P_std=1.0, t_eps=0.02, noise_scale=1.0,
        sampling_method="euler",
        time_shift_scale=1.0
        ):
        super(FlowLoss, self).__init__()

        self.in_channels = target_channels
        # Build net_kwargs: pass z_channels if provided and not already in net_kwargs
        _net_kwargs = dict(net_kwargs)
        _net_kwargs["in_channels"] = target_channels
        if z_channels is not None and "z_channels" not in _net_kwargs:
            _net_kwargs["z_channels"] = z_channels
        self.net = eval(net_class)(**_net_kwargs)
        # SimpleMLPAdaLN(
        #     in_channels=target_channels,
        #     model_channels=width,
        #     out_channels=target_channels,  # x-prediction
        #     z_channels=z_channels,
        #     num_res_blocks=depth,
        #     grad_checkpointing=grad_checkpointing
        # )

        # flow matching params
        self.P_mean = P_mean
        self.P_std = P_std
        self.t_eps = t_eps
        self.noise_scale = noise_scale

        # sampling params
        self.time_shift_scale = time_shift_scale
        self.num_sampling_steps = int(num_sampling_steps)
        self.sampling_method = sampling_method

    def sample_t(self, n: int, device=None):
        """Sample timesteps from logit-normal distribution"""
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, target, z, mask=None):
        """
        Flow matching loss with x-prediction.
        :param target: [N, C] target token latents
        :param z: [N, D] conditioning from AR transformer
        :param mask: optional mask for loss
        """
        bsz = target.shape[0]
        t = self.sample_t(bsz, device=target.device).view(-1, 1)
        e = torch.randn_like(target) * self.noise_scale
        
        # interpolate: z_t = t * x + (1-t) * noise
        z_t = t * target + (1 - t) * e
        # target velocity
        v = (target - z_t) / (1 - t).clamp_min(self.t_eps)

        # x-prediction
        x_pred = self.net(z_t, t.flatten(), z)
        # predicted velocity from x-prediction
        v_pred = (x_pred - z_t) / (1 - t).clamp_min(self.t_eps)

        # velocity loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=-1)

        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()

        return loss

    @torch.no_grad()
    def sample(self, z, temperature=1.0, cfg=1.0):
        """
        Generate samples using ODE integration.
        :param z: [N, D] conditioning from AR transformer.
                  When cfg != 1.0, z is already doubled as [z_cond, z_uncond].
        :param temperature: noise temperature
        :param cfg: classifier-free guidance scale
        """
        device = z.device

        if cfg != 1.0:
            # z is [z_cond, z_uncond], use same noise for both halves
            half_bsz = z.shape[0] // 2
            noise = temperature * self.noise_scale * torch.randn(half_bsz, self.in_channels, device=device)
            x = torch.cat([noise, noise], dim=0)
            bsz = z.shape[0]
        else:
            bsz = z.shape[0]
            x = temperature * self.noise_scale * torch.randn(bsz, self.in_channels, device=device)

        # timesteps from 0 to 1
        timesteps = torch.linspace(0.0, 1.0, self.num_sampling_steps + 1, device=device)
        timesteps = timesteps / (self.time_shift_scale - (self.time_shift_scale - 1) * timesteps)
        if self.sampling_method == "euler":
            stepper = self._euler_step
        elif self.sampling_method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError(f"Unknown sampling method: {self.sampling_method}")

        # ODE integration
        for i in range(self.num_sampling_steps - 1):
            t = timesteps[i].expand(bsz)
            t_next = timesteps[i + 1].expand(bsz)
            x = stepper(x, t, t_next, z, cfg)

        # last step with euler
        t = timesteps[-2].expand(bsz)
        t_next = timesteps[-1].expand(bsz)
        x = self._euler_step(x, t, t_next, z, cfg)

        return x

    @torch.no_grad()
    def _forward_sample(self, x, t, z, cfg=1.0):
        """
        Forward pass for sampling with optional CFG.
        When cfg != 1.0, x and z are already doubled as [cond, uncond].
        """
        t_input = t.view(-1, 1)

        # x-prediction
        x_pred = self.net(x, t, z)
        v_pred = (x_pred - x) / (1.0 - t_input).clamp_min(self.t_eps)

        if cfg != 1.0:
            # split cond/uncond predictions
            v_cond, v_uncond = v_pred.chunk(2, dim=0)
            # cfg interpolation
            v_cfg = v_uncond + cfg * (v_cond - v_uncond)
            # duplicate back for both halves
            v_pred = torch.cat([v_cfg, v_cfg], dim=0)

        return v_pred

    @torch.no_grad()
    def _euler_step(self, x, t, t_next, z, cfg=1.0):
        """Euler integration step"""
        v_pred = self._forward_sample(x, t, z, cfg)
        t_input = t.view(-1, 1)
        t_next_input = t_next.view(-1, 1)
        x_next = x + (t_next_input - t_input) * v_pred
        return x_next

    @torch.no_grad()
    def _heun_step(self, x, t, t_next, z, cfg=1.0):
        """Heun integration step (2nd order)"""
        t_input = t.view(-1, 1)
        t_next_input = t_next.view(-1, 1)

        # first euler step
        v_pred_t = self._forward_sample(x, t, z, cfg)
        x_next_euler = x + (t_next_input - t_input) * v_pred_t

        # evaluate at next point
        v_pred_t_next = self._forward_sample(x_next_euler, t_next, z, cfg)

        # average velocities
        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        x_next = x + (t_next_input - t_input) * v_pred
        return x_next


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
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
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
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
        t = t * 1000.0  # scale to [0, 1000] for better frequency coverage
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

# reference
# class JiTBlock(nn.Module):
#     def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
#         super().__init__()
#         self.norm1 = RMSNorm(hidden_size, eps=1e-6)
#         self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
#                               attn_drop=attn_drop, proj_drop=proj_drop)
#         self.norm2 = RMSNorm(hidden_size, eps=1e-6)
#         mlp_hidden_dim = int(hidden_size * mlp_ratio)
#         self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size, 6 * hidden_size, bias=True)
#         )

#     @torch.compile
#     def forward(self, x,  c, feat_rope=None):
#         shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
#         x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
#         x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
#         return x



def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype, device=query.device)

    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope is not None:
            q = rope(q)
            k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
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


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PiTBlock(nn.Module):
    def __init__(self, hidden_size, attn_dim, num_heads, cond_dim=None, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn_dim = attn_dim
        self.cond_dim = cond_dim if cond_dim is not None else hidden_size

        self.norm1 = RMSNorm(hidden_size, eps=1e-6)

        # compress/expand for bottleneck attention
        self.compress = nn.Linear(hidden_size, attn_dim)
        self.expand = nn.Linear(attn_dim, hidden_size)

        # attention in lower dim
        self.attn = Attention(attn_dim, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)

        # mlp in original dim
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

        # adaLN modulation (6 params: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.cond_dim, 6 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c, feat_rope=None):
        '''
        x: [N, L, hidden_size] (p**2 * D_pix)
        c: [N, D_cond] conditioning (t + c combined)
        '''
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

        # compress to attention dim
        def compact_attn(_x):
            _x = self.compress(_x)
            _x = self.attn(_x, feat_rope)
            _x = self.expand(_x)
            return _x
        x = x + gate_msa.unsqueeze(1) * compact_attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x



class PiTFinalLayer(nn.Module):
    """Final layer for PiT with adaLN modulation."""
    def __init__(self, hidden_size, out_channels, cond_dim=None):
        super().__init__()
        self.cond_dim = cond_dim if cond_dim is not None else hidden_size
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.cond_dim, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimplePiT(nn.Module):
    """
    Patch-wise image Transformer for diffusion/flow loss.
    :param in_channels: channels in the input Tensor (vae_embed_dim * patch_size^2 from MAR).
    :param z_channels: channels in the condition from AR transformer.
    :param depth: number of transformer blocks.
    """

    def __init__(
        self,
        in_channels,
        z_channels,
        depth,
        grad_checkpointing=False,

        in_pix_channels=16,
        patch_size=2,
        pix_embed_dim=256,
        pix_attn_dim=512,

        cond_dim=1024,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        in_context_len=32,
        in_context_start=8,
        pixelwise_adaln=False,
    ):
        super().__init__()

        self.pixelwise_adaln = pixelwise_adaln
        self.in_channels = in_channels
        self.in_pix_channels = in_pix_channels
        self.patch_size = patch_size
        self.pix_embed_dim = pix_embed_dim
        self.pix_attn_dim = pix_attn_dim
        self.cond_dim = cond_dim
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.grad_checkpointing = grad_checkpointing

        # compute spatial size from in_channels
        # in_channels = in_pix_channels * h * w (flattened image)
        self.spatial_size = int(math.sqrt(in_channels // in_pix_channels))
        self.num_patches = (self.spatial_size // patch_size) ** 2
        self.patch_dim = pix_embed_dim * patch_size * patch_size

        # conditioning embeddings
        self.c_embed = nn.Linear(z_channels, cond_dim)
        self.t_embed = TimestepEmbedder(cond_dim)

        # pixel embedding (conv to embed each pixel)
        self.x_embed = nn.Conv2d(in_pix_channels, pix_embed_dim, kernel_size=1, stride=1, bias=True)

        # positional embedding for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.patch_dim), requires_grad=False)

        # in-context tokens
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, self.patch_dim))
            nn.init.normal_(self.in_context_posemb, std=0.02)

        # RoPE for attention
        half_head_dim = pix_attn_dim // num_heads // 2
        hw_seq_len = self.spatial_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len
        )

        # transformer blocks
        self.blocks = nn.ModuleList([
            PiTBlock(
                hidden_size=self.patch_dim,
                attn_dim=pix_attn_dim,
                num_heads=num_heads,
                cond_dim=cond_dim,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0
            )
            for i in range(depth)
        ])

        # final layer: output same as input channels
        self.final_layer = PiTFinalLayer(self.patch_dim, self.patch_dim, cond_dim=cond_dim)

        # projection for in-context tokens (cond_dim -> patch_dim)
        if self.in_context_len > 0:
            self.in_context_proj = nn.Linear(cond_dim, self.patch_dim)

        # output projection (pix_embed_dim -> in_pix_channels)
        self.out_proj = nn.Conv2d(pix_embed_dim, in_pix_channels, kernel_size=1, stride=1, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_embed with 2D sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.patch_dim, int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Zero-out output projection
        nn.init.constant_(self.out_proj.weight, 0)
        nn.init.constant_(self.out_proj.bias, 0)

    def patchify(self, x):
        """
        x: [N, C, H, W] -> [N, L, c * p^2]
        """
        N, C, H, W = x.shape
        p = self.patch_size
        h_, w_ = H // p, W // p

        x = x.reshape(N, C, h_, p, w_, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [N, h_, w_, C, p, p]
        x = x.reshape(N, h_ * w_, C * p * p)
        return x

    def unpatchify(self, x):
        """
        x: [N, L, c * p^2 ] -> [N, C, H, W]
        """
        N = x.shape[0]
        p = self.patch_size
        C = self.pix_embed_dim
        h_ = w_ = int(self.num_patches ** 0.5)

        x = x.reshape(N, h_, w_, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)  # [N, C, h_, p, w_, p]
        x = x.reshape(N, C, h_ * p, w_ * p)
        return x

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        N, C = x.shape
        h = w = int(math.sqrt(C // self.in_pix_channels))
        x = x.reshape(N, self.in_pix_channels, h, w)

        c = self.c_embed(c)  # [N, D_cond]
        t = self.t_embed(t)  # [N, D_cond]
        c_adaln = c + t

        x = self.x_embed(x)  # [N, D_pix, h, w]
        x = self.patchify(x)  # [N, L, D_pix * p**2] (L = h//p * w//p)
        x = x + self.pos_embed

        in_context_added = False
        for i, block in enumerate(self.blocks):
            # in-context
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = self.in_context_proj(c).unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens = in_context_tokens + self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)
                in_context_added = True

            rope = self.feat_rope if not in_context_added else self.feat_rope_incontext
            if self.grad_checkpointing and self.training:
                x = checkpoint(block, x, c_adaln, rope, use_reentrant=False)
            else:
                x = block(x, c_adaln, rope)

        if in_context_added:
            x = x[:, self.in_context_len:]

        x = self.final_layer(x, c_adaln)
        x = self.unpatchify(x)  # [N, pix_embed_dim, h, w]
        x = self.out_proj(x)  # [N, in_pix_channels, h, w]

        x = x.reshape(N, C)

        
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
import numpy as np

from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm


def modulate_seq(x, shift, scale):
    """Modulate for sequence input: x is [N, L, D], shift/scale are [N, D]"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def modulate(x, shift, scale):
    return x * scale + shift

def precise_modulate(x, shift, scale):
    """Modulate with per-position parameters: x is [N, L, D], shift/scale are [N, L*D]"""
    N, L, D = x.shape
    shift = shift.view(N, L, D)
    scale = scale.view(N, L, D)
    return x * (1 + scale) + shift

class FlowLoss(nn.Module):
    """Flow Matching Loss with x-prediction (JIT formulation)"""
    def __init__(self,
        target_channels, num_sampling_steps,
        z_channels=None,  # Accept z_channels for MAR interface compatibility
        net_class = 'SimpleMLPAdaLN',
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
        if z_channels is not None :
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

class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
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
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0., use_fused_attn=True):
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

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.use_fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SimpleTransformerBlock(nn.Module):
    """
    Transformer block with AdaLN conditioning.
    :param d_model: model/feature dimension.
    :param num_heads: number of attention heads.
    :param d_cond: conditioning dimension (can differ from d_model).
    :param mlp_ratio: MLP hidden dim multiplier.
    :param use_fused_attn: whether to use F.scaled_dot_product_attention.
    """
    def __init__(self, d_model, num_heads, d_cond, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, use_fused_attn=True,
        precise_adaln = False,
        seq_len = None,
    ):
        super().__init__()
        self.precise_adaln = precise_adaln
        self.seq_len = seq_len
        self.d_model = d_model
        self.norm1 = RMSNorm(d_model, eps=1e-6)
        self.attn = Attention(d_model, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop, use_fused_attn=use_fused_attn)
        self.norm2 = RMSNorm(d_model, eps=1e-6)
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = SwiGLUFFN(d_model, mlp_hidden_dim, drop=proj_drop)
        adaln_dim = 6 * d_model if not precise_adaln else  6 * d_model *seq_len
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_cond, adaln_dim, bias=True)
        )

    # @torch.compile
    def forward(self, x, c):
        """
        :param x: [N, L, D] sequence of tokens.
        :param c: [N, d_cond] conditioning vector.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        if self.precise_adaln:
            N, L, D = x.shape
            gate_msa = gate_msa.view(N, L, D)
            gate_mlp = gate_mlp.view(N, L, D)
            x = x + gate_msa * self.attn(precise_modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp * self.mlp(precise_modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate_seq(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate_seq(self.norm2(x), shift_mlp, scale_mlp))
        return x


class SimpleTransformerFinalLayer(nn.Module):
    """
    Final layer with AdaLN for SimpleTransformer.
    """
    def __init__(self, d_model, out_dim, d_cond,
        precise_adaln = False,
        seq_len = None,
    ):
        super().__init__()
        self.precise_adaln = precise_adaln
        self.norm_final = RMSNorm(d_model, eps=1e-6)
        self.linear = nn.Linear(d_model, out_dim, bias=True)
        adaln_dim = 2 * d_model if not precise_adaln else 2 * d_model * seq_len
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_cond, adaln_dim, bias=True)
        )
    # @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        if self.precise_adaln:
            x = precise_modulate(self.norm_final(x), shift, scale)
        else:
            x = modulate_seq(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleTransformer(nn.Module):
    """
    Patch-wise Transformer for diffusion/flow loss with AdaLN conditioning.
    :param in_channels: channels in the input Tensor (vae_embed_dim * num_patches from MAR).
    :param z_channels: channels in the condition from AR transformer.
    :param d_model: hidden dimension of the transformer.
    :param d_cond: conditioning dimension (defaults to d_model if None).
    :param depth: number of transformer blocks.
    :param num_heads: number of attention heads.
    :param patch_size: patch size for internal patchification.
    :param vae_embed_dim: VAE embedding dimension per spatial location.
    :param mlp_ratio: MLP hidden dim multiplier.
    :param grad_checkpointing: whether to use gradient checkpointing.
    :param use_fused_attn: whether to use F.scaled_dot_product_attention.
    """

    def __init__(
        self,
        in_channels,
        z_channels,
        d_model=512,
        d_cond=None,
        depth=6,
        num_heads=8,
        patch_size=1,
        vae_embed_dim=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        grad_checkpointing=False,
        use_fused_attn=True,
        residual_cond = False,
        precise_adaln = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.d_cond = d_cond if d_cond is not None else d_model
        self.depth = depth
        self.patch_size = patch_size
        self.vae_embed_dim = vae_embed_dim
        self.grad_checkpointing = grad_checkpointing
        self.residual_cond = residual_cond
        self.precise_adaln = precise_adaln

        # Compute spatial dimensions
        # in_channels = vae_embed_dim * H * W where H, W are spatial dims of VAE latent
        self.num_spatial = in_channels // vae_embed_dim
        self.hw = int(math.sqrt(self.num_spatial))
        assert self.hw * self.hw == self.num_spatial, "in_channels must be vae_embed_dim * H * W with H=W"

        # Patch embedding: project patches to d_model
        self.num_patches = (self.hw // patch_size) ** 2
        patch_dim = vae_embed_dim * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, d_model)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

        # Time and condition embeddings -> d_cond
        self.time_embed = TimestepEmbedder(self.d_cond)
        self.cond_embed = nn.Linear(z_channels, self.d_cond)
        if self.residual_cond:
            self.cond_residual_proj = nn.Linear(z_channels, d_model )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(d_model, num_heads, self.d_cond, mlp_ratio, attn_drop, proj_drop, use_fused_attn,
                precise_adaln=precise_adaln, seq_len=self.num_patches)
            for _ in range(depth)
        ])

        # Final layer
        self.final_layer = SimpleTransformerFinalLayer(d_model, patch_dim, self.d_cond,
            precise_adaln=precise_adaln, seq_len=self.num_patches)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize positional embedding
        pos_embed = get_2d_sincos_pos_embed(self.d_model, int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

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

    def patchify(self, x):
        """
        :param x: [N, C, H, W] where C=vae_embed_dim
        :return: [N, L, patch_dim] where L=num_patches, patch_dim=vae_embed_dim*p*p
        """
        N, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0
        h, w = H // p, W // p
        x = x.reshape(N, C, h, p, w, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(N, h * w, C * p * p)
        return x

    def unpatchify(self, x):
        """
        :param x: [N, L, patch_dim] where L=num_patches
        :return: [N, C, H, W] where C=vae_embed_dim
        """
        N, L, _ = x.shape
        p = self.patch_size
        h = w = int(L ** 0.5)
        C = self.vae_embed_dim
        x = x.reshape(N, h, w, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(N, C, h * p, w * p)
        return x

    # @torch.compile
    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N, C] Tensor of inputs (C = vae_embed_dim * H * W).
        :param t: a 1-D batch of timesteps.
        :param c: [N, z_channels] conditioning from AR transformer.
        :return: an [N, C] Tensor of outputs.
        """
        N, C = x.shape

        # Reshape to spatial: [N, vae_embed_dim, H, W]
        x = x.reshape(N, self.vae_embed_dim, self.hw, self.hw)

        # Patchify: [N, L, patch_dim]
        x = self.patchify(x)

        # Project patches to d_model
        x = self.patch_embed(x)
        x = x + self.pos_embed
        if self.residual_cond:
            x = x + self.cond_residual_proj(c).unsqueeze(1)


        # Compute conditioning: time + AR condition -> d_cond
        t_emb = self.time_embed(t)
        c_emb = self.cond_embed(c)
        cond = t_emb + c_emb

        # Transformer blocks with AdaLN
        if self.grad_checkpointing and self.training and not torch.jit.is_scripting():
            for block in self.blocks:
                x = checkpoint(block, x, cond, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x, cond)

        # Final layer
        x = self.final_layer(x, cond)

        # Unpatchify: [N, vae_embed_dim, H, W]
        x = self.unpatchify(x)

        # Flatten back: [N, C]
        x = x.reshape(N, C)

        return x

class FullTransformer(nn.Module):
    def forward(self, x, t, c, attn_mask=None):
        """
        Apply the model to an input batch.
        :param x: an [N, C] Tensor of inputs, N = batch_size * L,  (C = vae_embed_dim * H * W).
        :param t: a 1-D batch of timesteps.
        :param c: [N, z_channels] conditioning from AR transformer.
        :return: an [N, C] Tensor of outputs.
        """
        N, C = x.shape
        L = ...
        B = N // L # it may include diffusion_batch_mul
        
        x = x.reshape(B, L, -1)
        c = c.reshape(B, L, -1)

        x = self.x_embedder(x)
        c = self.c_embed(c)
        x = x + c
        for block in self.blocks:
            x = block(x, c, attn_mask=attn_mask)
        x = self.final_layer(x, c)
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

    # def __init__(
    #     self,
    #     in_channels,
    #     model_channels,
    #     out_channels,
    #     z_channels,
    #     num_res_blocks,
    #     grad_checkpointing=False
    # ):
    def __init__(
        self,
        in_channels,
        z_channels,
        d_model,
        depth,
        out_channels=None,
        grad_checkpointing=False
    ):
        super().__init__()
        model_channels = d_model
        num_res_blocks = depth
        out_channels = out_channels if out_channels is not None else in_channels

        
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

    @torch.compile
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


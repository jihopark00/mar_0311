from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import Block

from models.diffloss import DiffLoss
from models.flowloss import FlowLoss, SimpleMLPAdaLN
from models.lora import set_ssl_encoder_mode
from util.model_util import get_2d_sincos_pos_embed
from einops import rearrange

def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking

def resize_image_per_patch(x, patch_size, target_patch_size, interpolate_mode='nearest', ):
    b = x.size(0)
    tok_h = x.size(2) // patch_size
    tok_w = x.size(3) // patch_size
    x = rearrange(x, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1=patch_size, p2=patch_size)
    # 2) interpolate per patch
    x = nn.functional.interpolate(x, size=target_patch_size, mode=interpolate_mode)
    # 3) unpatchify
    x = rearrange(x, '(b h w) c p1 p2 -> b c (h p1) (w p2)', b=b, h=tok_h, w=tok_w)
    return x

class MARSSL_Latent(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, 
        sslenc ='dinov2_vitg14_reg',
        sslenc_class_embed_num = 32,
        sslenc_class_embed_start_layer = 10,
        sslenc_lora_rank = 8,
        sslenc_lora_alpha = 16,
        sslenc_block_start = 0,
        sslenc_preenc_embed_dim = 1024,
        sslenc_preenc_depth = 0,
        sslenc_preenc_num_heads = 16,
        
        img_size=256, vae_stride=1, patch_size=16,
        #  encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4., norm_layer=nn.LayerNorm,
        vae_embed_dim=16,
        mask_ratio_min=0.7,
        label_drop_prob=0.1,
        class_num=1000,
        attn_dropout=0.1,
        proj_dropout=0.1,
        #  buffer_size=64,
        #  diffloss_d=3,
        #  diffloss_w=1024,
        #  num_sampling_steps='100',
        diffusion_batch_mul=4,
        grad_checkpointing=False,

        diffloss_class = "DiffLoss",
        diffloss_kwargs = {
            "width": 1024,
            "depth": 3,
            "num_sampling_steps": '100',
            "grad_checkpointing": False,
                 }
                 ):
        super().__init__()

        sslenc = torch.hub.load('facebookresearch/dinov2', sslenc, pretrained=True) # TODO:check if it's weight is not init
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

        self.img_size_ssl = self.seq_h * sslenc.patch_size

        encoder_embed_dim = sslenc.embed_dim
        self.encoder_embed_dim = encoder_embed_dim
        # reset sslenc blocks
        self.sslenc_block_start = sslenc_block_start
        sslenc.blocks = sslenc.blocks[sslenc_block_start:]
        
        self.sslenc_preenc_depth = sslenc_preenc_depth
        sslenc_preenc_pos_embed = get_2d_sincos_pos_embed(sslenc_preenc_embed_dim, self.seq_h)
        self.sslenc_preenc_pos_embed = nn.Parameter(torch.from_numpy(sslenc_preenc_pos_embed).float().unsqueeze(0), requires_grad=False)
        self.sslenc_preenc_mask_token = nn.Parameter(torch.zeros(1, 1, sslenc_preenc_embed_dim))
        self.sslenc_preenc_proj_in = nn.Linear(self.token_embed_dim, sslenc_preenc_embed_dim, bias=True) 
        self.sslenc_preenc_blocks = nn.ModuleList([
            Block(sslenc_preenc_embed_dim, sslenc_preenc_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(sslenc_preenc_depth)])
        self.sslenc_preenc_norm = norm_layer(sslenc_preenc_embed_dim)
        self.sslenc_enc_proj_out = nn.Linear(sslenc_preenc_embed_dim, encoder_embed_dim, bias=True)
        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, sslenc_class_embed_num * encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))
        
        self.class_emb_pos_embed = nn.Parameter(torch.zeros(1, sslenc_class_embed_num, encoder_embed_dim))
        
        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        # self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        # self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        # self.buffer_size = buffer_size
        # self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        # self.encoder_blocks = nn.ModuleList([
        #     Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
        #           proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        # self.encoder_norm = norm_layer(encoder_embed_dim)
        self.buffer_size = sslenc.num_register_tokens + 1 + sslenc_class_embed_num
        self.sslenc_class_embed_num = sslenc_class_embed_num
        self.sslenc_class_embed_start_layer = sslenc_class_embed_start_layer
        
        # --------------------------------------------------------------------------
        # MAR decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))

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
            "z_channels": decoder_embed_dim
        })
        diffloss_class = eval(diffloss_class)
        self.diffloss = diffloss_class(**diffloss_kwargs)
        self.diffusion_batch_mul = diffusion_batch_mul

        # SSL Encoder
        self.sslenc = sslenc

        # apply lora
        self.sslenc_lora_rank = sslenc_lora_rank
        self.sslenc_lora_alpha = sslenc_lora_alpha
        if sslenc_lora_rank > 0:
            set_ssl_encoder_mode(self.sslenc, mode="lora", lora_r=sslenc_lora_rank, lora_alpha=sslenc_lora_alpha)
        else:
            set_ssl_encoder_mode(self.sslenc, mode="freeze")

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.sslenc_preenc_mask_token, std=.02)
        torch.nn.init.normal_(self.class_emb_pos_embed, std=.02)
        # torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

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
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask
    
    def sslenc_forward(self, x, masks, class_embedding): # x is patchified latent tokens
        bsz = x.size(0)
        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).unsqueeze(-1).cuda().to(x.dtype)
            fake_latent = self.fake_latent.unsqueeze(0).expand(bsz, self.sslenc_class_embed_num, -1)
            class_embedding = drop_latent_mask * fake_latent + (1 - drop_latent_mask) * class_embedding

        sslenc = self.sslenc
        masks = masks.bool()

        # pre-encode: project patchified latent tokens to encoder_embed_dim
        x = self.sslenc_preenc_proj_in(x)
        if self.sslenc_preenc_depth > 0:
            if masks is not None:
                x = torch.where(masks.unsqueeze(-1), self.sslenc_preenc_mask_token.expand(x.shape[0], x.shape[1], -1).to(x.dtype), x)
            x = x + self.sslenc_preenc_pos_embed.to(x.dtype)

            if self.grad_checkpointing and not torch.jit.is_scripting():
                def _forward(blk, x):
                    return checkpoint(blk, x, use_reentrant=False)
            else:
                def _forward(blk, x):
                    return blk(x)

            for blk in self.sslenc_preenc_blocks:
                x = _forward(blk, x)
        x = self.sslenc_preenc_norm(x)
        x = self.sslenc_enc_proj_out(x)

        # apply mask
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), sslenc.mask_token.to(x.dtype).unsqueeze(0), x)

        # add cls token and pos encoding
        x = torch.cat((sslenc.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        w = h = self.img_size_ssl
        x = x + sslenc.interpolate_pos_encoding(x, w, h)

        # add register tokens
        if sslenc.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    sslenc.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        # forward through sslenc blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            def _forward(blk, x):
                return checkpoint(blk, x, use_reentrant=False)
        else:
            def _forward(blk, x):
                return blk(x)
        for i, blk in enumerate(sslenc.blocks):
            if self.sslenc_class_embed_num > 0 and i == self.sslenc_class_embed_start_layer:
                class_embedding = class_embedding + self.class_emb_pos_embed
                x = torch.cat([class_embedding, x], dim=1)
            x = _forward(blk, x)

        x = sslenc.norm(x)
        return x

    def forward_mae_decoder(self, x, mask):

        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed_learned
        # mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # # pad mask tokens
        # mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        # x_after_pad = mask_tokens.clone()
        # x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        # decoder position embedding
        # x = x_after_pad + self.decoder_pos_embed_learned

        # residual additon to mask
        buffer, x = x[:, :self.buffer_size], x[:, self.buffer_size:]
        mask_token = mask.unsqueeze(-1) * self.mask_token.to(x.dtype)
        x = x + mask_token
        x = torch.cat([buffer, x], dim=1)

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.decoder_blocks:
                x = block(x)
        x = self.decoder_norm(x)

        x = x[:, self.buffer_size:]
        x = x + self.diffusion_pos_embed_learned
        return x

    def forward_loss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def forward(self, imgs, labels):

        # class embed
        class_embedding = self.class_emb(labels).view(-1, self.sslenc_class_embed_num, self.encoder_embed_dim)
        
        # patchify and mask (drop) tokens
        x = self.patchify(imgs)
        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz=gt_latents.size(0))
        mask = self.random_masking(gt_latents, orders)

        # mae encoder
        x = self.sslenc_forward(x, mask, class_embedding)

        # mae decoder
        z = self.forward_mae_decoder(x, mask)

        # diffloss
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False):
        grad_checkpointing = self.grad_checkpointing  # Enable grad checkpointing during sampling for memory efficiency
        diffloss_grad_checkpointing = self.diffloss.net.grad_checkpointing

        self.grad_checkpointing = False  # Disable grad checkpointing during sampling for speed
        self.diffloss.net.grad_checkpointing = False
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
                class_embedding = self.class_emb(labels).view(-1, self.sslenc_class_embed_num, self.encoder_embed_dim)
            else:
                class_embedding = self.fake_latent.unsqueeze(0).expand(bsz, self.sslenc_class_embed_num, -1)
            if not cfg == 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                fake_class_embedding = self.fake_latent.unsqueeze(0).expand(bsz, self.sslenc_class_embed_num, -1)
                class_embedding = torch.cat([class_embedding, fake_class_embedding], dim=0)
                mask = torch.cat([mask, mask], dim=0)

            # mae encoder
            x = self.sslenc_forward(tokens, mask, class_embedding)

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
        
        self.grad_checkpointing = grad_checkpointing  # Restore original grad checkpointing setting
        self.diffloss.net.grad_checkpointing = diffloss_grad_checkpointing
        return tokens

def mar_ssl_latent(**kwargs):
    model = MARSSL_Latent(**kwargs)
    return model

# def mar_base(**kwargs):
#     model = MAR(
#         encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
#         decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def mar_large(**kwargs):
#     model = MAR(
#         encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
#         decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def mar_huge(**kwargs):
#     model = MAR(
#         encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
#         decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

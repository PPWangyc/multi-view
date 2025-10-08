# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from masks.utils import apply_masks
from utils.log_utils import get_logger
from utils.tensor_utils import repeat_interleave_batch, trunc_normal_

logger = get_logger()

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid length
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))

        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ConvEmbed(nn.Module):
    """
    3x3 Convolution stems for ViT following ViTC models
    """

    def __init__(self, channels, strides, img_size=224, in_chans=3, batch_norm=True):
        super().__init__()
        # Build the stems
        stem = []
        channels = [in_chans] + channels
        for i in range(len(channels) - 2):
            stem += [nn.Conv2d(channels[i], channels[i+1], kernel_size=3,
                               stride=strides[i], padding=1, bias=(not batch_norm))]
            if batch_norm:
                stem += [nn.BatchNorm2d(channels[i+1])]
            stem += [nn.ReLU(inplace=True)]
        stem += [nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=strides[-1])]
        self.stem = nn.Sequential(*stem)

        # Comptute the number of patches
        stride_prod = int(np.prod(strides))
        self.num_patches = (img_size[0] // stride_prod)**2

    def forward(self, x):
        p = self.stem(x)
        return p.flatten(2).transpose(1, 2)

class VisionTransformerPredictor(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        num_patches,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # --
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim),
                                                requires_grad=False)
        predictor_pos_embed = get_2d_sincos_pos_embed(self.predictor_pos_embed.shape[-1],
                                                      int(num_patches**.5),
                                                      cls_token=False)
        self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))
        # --
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks_x, masks):
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        # -- Batch Size
        B = len(x) // len(masks_x)

        # -- map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x)

        # -- add positional embedding to x tokens
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_x)

        _, N_ctxt, D = x.shape

        # -- concat mask tokens to x
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        # --
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        # --
        pred_tokens += pos_embs
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x

class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=12,
        predictor_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        # --
        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # --
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                            int(self.patch_embed.num_patches**.5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks=None):
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        # -- patchify x
        x = self.patch_embed(x)
        B, N, D = x.shape

        # -- add positional embedding to x
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed

        # -- mask x
        if masks is not None:
            x = apply_masks(x, masks)

        # -- fwd prop
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        if self.norm is not None:
            x = self.norm(x)

        return x

    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)

class IJEPA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = vit_base()
        self.target_encoder = copy.deepcopy(self.encoder)

        self.predictor = vit_predictor(
            num_patches=self.encoder.patch_embed.num_patches,
            embed_dim=self.encoder.embed_dim,
            predictor_embed_dim=config['model']['model_params']['predictor_embed_dim'],
            depth=config['model']['model_params']['predictor_depth'],
            num_heads=self.encoder.num_heads,
        )
        # ijepa specific params
        self.allow_overlap = config['model']['model_params']['allow_overlap']
        self.height, self.width = config['model']['model_params']['image_size'] // config['model']['model_params']['patch_size'], config['model']['model_params']['image_size'] // config['model']['model_params']['patch_size']
        self.npred = config['model']['model_params']['num_predictor_masks']
        self.min_keep = config['model']['model_params']['min_keep']
        self.nenc = config['model']['model_params']['num_encoder_masks']

        # -- momentum schedule
        ema = config['model']['model_params']['ema']
        ipe_scale = config['training']['ipe_scale']
        num_epochs = config['training']['num_epochs']
        ipe = config['training']['ipe']
        self.momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                            for i in range(int(ipe*num_epochs*ipe_scale)+1))

    @torch.no_grad()
    def forward_target(self, imgs, masks_enc, masks_pred):
        h = self.target_encoder(imgs)
        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
        B = len(h)
        # -- create targets (masked regions of h)
        h = apply_masks(h, masks_pred)
        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
        return h

    def forward_context(self, imgs, masks_enc, masks_pred):
        z = self.encoder(imgs, masks_enc)
        z = self.predictor(z, masks_enc, masks_pred)
        return z
    
    def sample_masks(self, x):
        B = len(x)
        g = torch.Generator()
        p_size = self._sample_block_size(
            generator=g,
            scale=(0.15, 0.2),
            aspect_ratio_scale=(0.75,1.5))
        e_size = self._sample_block_size(
            generator=g,
            scale=(0.85, 1.0),
            aspect_ratio_scale=(1., 1.))
        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width

        for _ in range(B):

            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions= None
            except Exception as e:
                logger.warning(f'Encountered exception in mask-generator {e}')

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        return collated_masks_enc, collated_masks_pred
    
    @torch.no_grad()
    def update_target(self):
        m = next(self.momentum_scheduler)
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.data)

    def forward(self, x):
        x = x['image']
        # -- sample masks
        masks_enc, masks_pred = self.sample_masks(x)
        # -- fwd prop
        h = self.forward_target(x, masks_enc, masks_pred)
        z = self.forward_context(x, masks_enc, masks_pred)
        loss = self.compute_loss(z, h)
        result_dict = {
            'loss': loss,
            'h': h,
            'z': z
        }
        return result_dict

    def compute_loss(
        self,
        z,
        h,
        **kwargs,
    ) -> tuple[torch.tensor, list[dict]]:
        loss = F.smooth_l1_loss(z, h)
        return loss

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        # --
        return mask, mask_complement

def vit_predictor(**kwargs):
    model = VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model

def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base(patch_size=16,  **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_huge(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_giant(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

VIT_EMBED_DIMS = {
    'vit_tiny': 192,
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
    'vit_huge': 1280,
    'vit_giant': 1408,
}

# from transformers import (
#     IJepaConfig, 
#     IJepaModel,
#     AutoModel,
#     ViTMAEModel,
#     ViTMAEConfig,
#     ViTModel,
#     ViTConfig,
#     AutoImageProcessor,
# )
# import torch
# import torch.nn as nn
# from utils.log_utils import get_logger
# from PIL import Image

# logger = get_logger()

# def transfer_vit_mae_to_ijepa(ijepa_config, vit_mae_model_name="facebook/vit-mae-base"):
#     """
#     Transfer weights from a ViT-MAE model to an IJEPA model.
    
#     Args:
#         ijepa_config (IJepaConfig): Configuration for IJEPA model
#         vit_mae_model_name (str): Name of the pretrained ViT-MAE model
    
#     Returns:
#         IJepaModel: IJEPA model with transferred weights
#     """
#     logger.info(f"Loading ViT-MAE model: {vit_mae_model_name}")
    
#     # Load ViT-MAE model and config
#     vit_mae_model = ViTMAEModel.from_pretrained(vit_mae_model_name)
#     vit_mae_config = ViTMAEConfig.from_pretrained(vit_mae_model_name)
    
#     logger.info(f"ViT-MAE config: hidden_size={vit_mae_config.hidden_size}, "
#           f"num_hidden_layers={vit_mae_config.num_hidden_layers}, "
#           f"num_attention_heads={vit_mae_config.num_attention_heads}")
    
#     logger.info(f"IJEPA config: hidden_size={ijepa_config.hidden_size}, "
#           f"num_hidden_layers={ijepa_config.num_hidden_layers}, "
#           f"num_attention_heads={ijepa_config.num_attention_heads}")
    
#     # Create IJEPA model
#     ijepa_model = IJepaModel(ijepa_config)
    
#     # Transfer weights
#     logger.info("Transferring weights from ViT-MAE to IJEPA...")
    
#     # Get state dicts
#     vit_mae_state_dict = vit_mae_model.state_dict()
#     ijepa_state_dict = ijepa_model.state_dict()
    
#     # Create mapping of parameter names
#     # The architectures should be compatible, so we can map most parameters directly
#     transferred_count = 0
#     skipped_count = 0
    
#     for ijepa_name, ijepa_param in ijepa_state_dict.items():
#         if ijepa_name in vit_mae_state_dict:
#             # Check if shapes match
#             if vit_mae_state_dict[ijepa_name].shape == ijepa_param.shape:
#                 ijepa_state_dict[ijepa_name] = vit_mae_state_dict[ijepa_name].clone()
#                 transferred_count += 1
#                 logger.info(f"Transferred: {ijepa_name}")
#             else:
#                 # Handle position embedding mismatch: MAE has CLS token, IJEPA doesn't
#                 if (ijepa_name == "embeddings.position_embeddings" and 
#                     vit_mae_state_dict[ijepa_name].shape[1] == ijepa_param.shape[1] + 1 and
#                     vit_mae_state_dict[ijepa_name].shape[0] == ijepa_param.shape[0] and
#                     vit_mae_state_dict[ijepa_name].shape[2] == ijepa_param.shape[2]):
#                     # Skip the first CLS token position embedding from MAE
#                     ijepa_state_dict[ijepa_name] = vit_mae_state_dict[ijepa_name][:, 1:, :].clone()
#                     transferred_count += 1
#                     logger.info(f"Transferred (skipped CLS): {ijepa_name}")
#                 else:
#                     logger.warning(f"Shape mismatch for {ijepa_name}: "
#                           f"ViT-MAE {vit_mae_state_dict[ijepa_name].shape} vs "
#                           f"IJEPA {ijepa_param.shape}")
#                     skipped_count += 1
#         else:
#             logger.warning(f"Parameter not found in ViT-MAE: {ijepa_name}")
#             skipped_count += 1
    
#     # Load the transferred state dict
#     ijepa_model.load_state_dict(ijepa_state_dict, strict=False)
    
#     logger.info(f"Weight transfer completed!")
#     logger.info(f"Transferred: {transferred_count} parameters")
#     logger.info(f"Skipped: {skipped_count} parameters")
    
#     return ijepa_model

# def transfer_dino_to_ijepa(ijepa_config, dino_model_name="facebook/dino-vitb16"):
#     """
#     Transfer weights from a DINO ViT-B-16 model to an IJEPA model.
    
#     Args:
#         ijepa_config (IJepaConfig): Configuration for IJEPA model
#         dino_model_name (str): Name of the pretrained DINO model
    
#     Returns:
#         IJepaModel: IJEPA model with transferred weights
#     """
#     logger.info(f"Loading DINO model: {dino_model_name}")
    
#     # Load DINO model and config
#     dino_model = ViTModel.from_pretrained(dino_model_name)
#     dino_config = ViTConfig.from_pretrained(dino_model_name)
    
#     logger.info(f"DINO config: hidden_size={dino_config.hidden_size}, "
#           f"num_hidden_layers={dino_config.num_hidden_layers}, "
#           f"num_attention_heads={dino_config.num_attention_heads}")
    
#     logger.info(f"IJEPA config: hidden_size={ijepa_config.hidden_size}, "
#           f"num_hidden_layers={ijepa_config.num_hidden_layers}, "
#           f"num_attention_heads={ijepa_config.num_attention_heads}")
    
#     # Create IJEPA model
#     ijepa_model = IJepaModel(ijepa_config)
    
#     # Transfer weights
#     logger.info("Transferring weights from DINO to IJEPA...")
    
#     # Get state dicts
#     dino_state_dict = dino_model.state_dict()
#     ijepa_state_dict = ijepa_model.state_dict()
    
#     # Create mapping of parameter names
#     # DINO and IJEPA should have compatible architectures
#     transferred_count = 0
#     skipped_count = 0
    
#     for ijepa_name, ijepa_param in ijepa_state_dict.items():
#         if ijepa_name in dino_state_dict:
#             # Check if shapes match
#             if dino_state_dict[ijepa_name].shape == ijepa_param.shape:
#                 ijepa_state_dict[ijepa_name] = dino_state_dict[ijepa_name].clone()
#                 transferred_count += 1
#                 logger.info(f"Transferred: {ijepa_name}")
#             else:
#                 # Handle position embedding mismatch: DINO has CLS token, IJEPA doesn't
#                 if (ijepa_name == "embeddings.position_embeddings" and 
#                     dino_state_dict[ijepa_name].shape[1] == ijepa_param.shape[1] + 1 and
#                     dino_state_dict[ijepa_name].shape[0] == ijepa_param.shape[0] and
#                     dino_state_dict[ijepa_name].shape[2] == ijepa_param.shape[2]):
#                     # Skip the first CLS token position embedding from DINO
#                     ijepa_state_dict[ijepa_name] = dino_state_dict[ijepa_name][:, 1:, :].clone()
#                     transferred_count += 1
#                     logger.info(f"Transferred (skipped CLS): {ijepa_name}")
#                 else:
#                     logger.warning(f"Shape mismatch for {ijepa_name}: "
#                           f"DINO {dino_state_dict[ijepa_name].shape} vs "
#                           f"IJEPA {ijepa_param.shape}")
#                     skipped_count += 1
#         else:
#             logger.warning(f"Parameter not found in DINO: {ijepa_name}")
#             skipped_count += 1
    
#     # Load the transferred state dict
#     ijepa_model.load_state_dict(ijepa_state_dict, strict=False)
    
#     logger.info(f"Weight transfer completed!")
#     logger.info(f"Transferred: {transferred_count} parameters")
#     logger.info(f"Skipped: {skipped_count} parameters")
    
#     return ijepa_model

# def main():
#     ijepa_config = IJepaConfig()
    
#     # Example usage for ViT-MAE transfer
#     logger.info("=== ViT-MAE to IJEPA Transfer ===")
#     model_ijepa = transfer_vit_mae_to_ijepa(ijepa_config)
#     logger.info(f"Final model type: {type(model_ijepa)}")
#     logger.info(f"Model config: {model_ijepa.config.model_type}")
#     logger.info("\n" + "="*50 + "\n")
    
#     # Example usage for DINO transfer
#     # logger.info("=== DINO to IJEPA Transfer ===")
#     # model_ijepa = transfer_dino_to_ijepa(ijepa_config)
#     # logger.info(f"Final model type: {type(model_ijepa)}")
#     # logger.info(f"Model config: {model_ijepa.config.model_type}")
#     # logger.info("\n" + "="*50 + "\n")

#     from datasets import load_dataset
#     dataset = load_dataset("huggingface/cats-image")
#     image = dataset["test"]["image"][0]
#     image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
#     inputs = image_processor(image, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model_ijepa(**inputs)
#     print(outputs['last_hidden_state'].shape)

# if __name__ == "__main__":
#     main()
from typing import Dict, Optional

import torch
import torch.nn as nn
import numpy as np
from jaxtyping import Float
from transformers import ViTMAEConfig, ViTMAEForPreTraining, ViTModel
from transformers.models.vit_mae.modeling_vit_mae import (ViTMAEDecoder,
                                                          ViTMAEDecoderOutput)
MAE_VIT_SMALL_PATH = "data/checkpoints/mae/vit-small/vit-small-patch16-224.pth"
def load_mae_ckpt(model_path, vit, decoder):
    ckpt = torch.load(model_path, map_location='cpu')['model']
    print(f'Loading MAE checkpoint from {model_path}...')
    
    vit_state_dict = vit.state_dict()
    new_state_dict = {}
    
    # Load patch embeddings
    if 'patch_embed.proj.weight' in ckpt:
        new_state_dict['embeddings.patch_embeddings.projection.weight'] = ckpt['patch_embed.proj.weight']
        new_state_dict['embeddings.patch_embeddings.projection.bias'] = ckpt['patch_embed.proj.bias']
    
    # Load position embeddings (excluding cls token position)
    if 'pos_embed' in ckpt:
        # MAE pos_embed includes cls token, ViT position_embeddings doesn't include it separately
        new_state_dict['embeddings.position_embeddings'] = ckpt['pos_embed']
    
    # Load cls token
    if 'cls_token' in ckpt:
        new_state_dict['embeddings.cls_token'] = ckpt['cls_token']
    
    # Load encoder blocks
    for i in range(12):  # 12 layers for ViT-Small
        mae_prefix = f'blocks.{i}'
        vit_prefix = f'encoder.layer.{i}'
        
        # Layer norm before attention
        new_state_dict[f'{vit_prefix}.layernorm_before.weight'] = ckpt[f'{mae_prefix}.norm1.weight']
        new_state_dict[f'{vit_prefix}.layernorm_before.bias'] = ckpt[f'{mae_prefix}.norm1.bias']
        
        # Attention: Split QKV into separate Q, K, V
        qkv_weight = ckpt[f'{mae_prefix}.attn.qkv.weight']
        qkv_bias = ckpt[f'{mae_prefix}.attn.qkv.bias']
        
        # Split into Q, K, V (each is 1/3 of the combined dimension)
        dim = qkv_weight.shape[0] // 3
        new_state_dict[f'{vit_prefix}.attention.attention.query.weight'] = qkv_weight[:dim]
        new_state_dict[f'{vit_prefix}.attention.attention.query.bias'] = qkv_bias[:dim]
        new_state_dict[f'{vit_prefix}.attention.attention.key.weight'] = qkv_weight[dim:2*dim]
        new_state_dict[f'{vit_prefix}.attention.attention.key.bias'] = qkv_bias[dim:2*dim]
        new_state_dict[f'{vit_prefix}.attention.attention.value.weight'] = qkv_weight[2*dim:]
        new_state_dict[f'{vit_prefix}.attention.attention.value.bias'] = qkv_bias[2*dim:]
        
        # Attention output projection
        new_state_dict[f'{vit_prefix}.attention.output.dense.weight'] = ckpt[f'{mae_prefix}.attn.proj.weight']
        new_state_dict[f'{vit_prefix}.attention.output.dense.bias'] = ckpt[f'{mae_prefix}.attn.proj.bias']
        
        # Layer norm after attention
        new_state_dict[f'{vit_prefix}.layernorm_after.weight'] = ckpt[f'{mae_prefix}.norm2.weight']
        new_state_dict[f'{vit_prefix}.layernorm_after.bias'] = ckpt[f'{mae_prefix}.norm2.bias']
        
        # MLP
        new_state_dict[f'{vit_prefix}.intermediate.dense.weight'] = ckpt[f'{mae_prefix}.mlp.fc1.weight']
        new_state_dict[f'{vit_prefix}.intermediate.dense.bias'] = ckpt[f'{mae_prefix}.mlp.fc1.bias']
        new_state_dict[f'{vit_prefix}.output.dense.weight'] = ckpt[f'{mae_prefix}.mlp.fc2.weight']
        new_state_dict[f'{vit_prefix}.output.dense.bias'] = ckpt[f'{mae_prefix}.mlp.fc2.bias']
    
    # Load final layer norm
    if 'norm.weight' in ckpt:
        new_state_dict['layernorm.weight'] = ckpt['norm.weight']
        new_state_dict['layernorm.bias'] = ckpt['norm.bias']
    
    # Load into ViT model (strict=False to allow missing pooler weights)
    missing_keys, unexpected_keys = vit.load_state_dict(new_state_dict, strict=False)
    print(f'ViT Missing keys: {missing_keys}')
    print(f'ViT Unexpected keys: {unexpected_keys}')
    
    # Load decoder weights
    decoder_state_dict = {}
    
    # Load decoder-specific tokens and embeddings
    if 'mask_token' in ckpt:
        decoder_state_dict['mask_token'] = ckpt['mask_token']
    
    if 'decoder_pos_embed' in ckpt:
        decoder_state_dict['decoder_pos_embed'] = ckpt['decoder_pos_embed']
    
    # Load decoder_view_embed if it exists in your decoder
    if 'decoder_view_embed' in ckpt:
        decoder_state_dict['decoder_view_embed'] = ckpt['decoder_view_embed']
    
    # Load decoder embedding projection
    if 'decoder_embed.weight' in ckpt:
        decoder_state_dict['decoder_embed.weight'] = ckpt['decoder_embed.weight']
        decoder_state_dict['decoder_embed.bias'] = ckpt['decoder_embed.bias']
    
    # Load decoder transformer blocks
    for i in range(8):  # 8 decoder layers for MAE
        mae_prefix = f'decoder_blocks.{i}'
        dec_prefix = f'decoder_layers.{i}'
        
        # Layer norm before attention
        decoder_state_dict[f'{dec_prefix}.layernorm_before.weight'] = ckpt[f'{mae_prefix}.norm1.weight']
        decoder_state_dict[f'{dec_prefix}.layernorm_before.bias'] = ckpt[f'{mae_prefix}.norm1.bias']
        
        # Attention: Split QKV into separate Q, K, V
        qkv_weight = ckpt[f'{mae_prefix}.attn.qkv.weight']
        qkv_bias = ckpt[f'{mae_prefix}.attn.qkv.bias']
        
        # Split into Q, K, V
        dim = qkv_weight.shape[0] // 3
        decoder_state_dict[f'{dec_prefix}.attention.attention.query.weight'] = qkv_weight[:dim]
        decoder_state_dict[f'{dec_prefix}.attention.attention.query.bias'] = qkv_bias[:dim]
        decoder_state_dict[f'{dec_prefix}.attention.attention.key.weight'] = qkv_weight[dim:2*dim]
        decoder_state_dict[f'{dec_prefix}.attention.attention.key.bias'] = qkv_bias[dim:2*dim]
        decoder_state_dict[f'{dec_prefix}.attention.attention.value.weight'] = qkv_weight[2*dim:]
        decoder_state_dict[f'{dec_prefix}.attention.attention.value.bias'] = qkv_bias[2*dim:]
        
        # Attention output projection
        decoder_state_dict[f'{dec_prefix}.attention.output.dense.weight'] = ckpt[f'{mae_prefix}.attn.proj.weight']
        decoder_state_dict[f'{dec_prefix}.attention.output.dense.bias'] = ckpt[f'{mae_prefix}.attn.proj.bias']
        
        # Layer norm after attention
        decoder_state_dict[f'{dec_prefix}.layernorm_after.weight'] = ckpt[f'{mae_prefix}.norm2.weight']
        decoder_state_dict[f'{dec_prefix}.layernorm_after.bias'] = ckpt[f'{mae_prefix}.norm2.bias']
        
        # MLP
        decoder_state_dict[f'{dec_prefix}.intermediate.dense.weight'] = ckpt[f'{mae_prefix}.mlp.fc1.weight']
        decoder_state_dict[f'{dec_prefix}.intermediate.dense.bias'] = ckpt[f'{mae_prefix}.mlp.fc1.bias']
        decoder_state_dict[f'{dec_prefix}.output.dense.weight'] = ckpt[f'{mae_prefix}.mlp.fc2.weight']
        decoder_state_dict[f'{dec_prefix}.output.dense.bias'] = ckpt[f'{mae_prefix}.mlp.fc2.bias']
    
    # Load decoder final layer norm
    if 'decoder_norm.weight' in ckpt:
        decoder_state_dict['decoder_norm.weight'] = ckpt['decoder_norm.weight']
        decoder_state_dict['decoder_norm.bias'] = ckpt['decoder_norm.bias']
    
    # Load decoder prediction head
    if 'decoder_pred.weight' in ckpt:
        decoder_state_dict['decoder_pred.weight'] = ckpt['decoder_pred.weight']
        decoder_state_dict['decoder_pred.bias'] = ckpt['decoder_pred.bias']
    
    if decoder is None:
        print(f'Skipping decoder loading')
        return vit, None
    # Load into decoder model
    missing_keys_dec, unexpected_keys_dec = decoder.load_state_dict(decoder_state_dict, strict=False)
    print(f'Decoder missing keys: {missing_keys_dec}')
    print(f'Decoder unexpected keys: {unexpected_keys_dec}')
    
    return vit, decoder
    
class MultiViewTransformer_OLD(torch.nn.Module):
    """Vision Transformer implementation."""

    def __init__(self, config):
        super().__init__()
        # Set up ViT architecture
        self.mask_ratio = config['model']['model_params']['mask_ratio']
        self.avail_views = config['data']['avail_views']
        self.num_views = len(self.avail_views)
        config['model']['model_params']['num_views'] = self.num_views
        self.config = ViTMAEConfig(**config['model']['model_params'])

        self.vit = ViTModel.from_pretrained(config['model']['pretrained'], use_safetensors=True)
        # fix sin-cos embedding
        self.vit.embeddings.position_embeddings.requires_grad = False
        
        h = w = config['model']['model_params']['image_size'] // config['model']['model_params']['patch_size']
        self.num_patches = h * w
        self.hidden_size = config['model']['model_params']['hidden_size']
        # self.view_embeddings = torch.nn.Parameter(torch.randn(1, self.num_views, self.num_patches, self.hidden_size))

        self.decoder = MultiViewTransformerDecoder(self.config, self.num_patches, self.num_views)
        # self.vit, self.decoder = load_mae_ckpt(MAE_VIT_SMALL_PATH, self.vit, self.decoder)

    def patchify(self, pixel_values):
        patch_size, num_channels = self.config.patch_size, self.config.decoder_num_channels
        # sanity checks
        if pixel_values.shape[2] != num_channels:
            raise ValueError(
                "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
            )

        # patchify
        batch_size = pixel_values.shape[0]
        num_views = pixel_values.shape[1]
        num_patches_h = pixel_values.shape[3] // patch_size
        num_patches_w = pixel_values.shape[4] // patch_size
        patchified_pixel_values = pixel_values.reshape(
            batch_size*num_views, num_channels, num_patches_h, patch_size, num_patches_w, patch_size
        )
        patchified_pixel_values = torch.einsum("nchpwq->nhwpqc", patchified_pixel_values)
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size, num_patches_h * num_patches_w * num_views, patch_size**2 * num_channels
        )
        return patchified_pixel_values

    def unpatchify(self, patchified_pixel_values, original_image_size: Optional[tuple[int, int]] = None):
        """
        Args:
            patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
            original_image_size (`tuple[int, int]`, *optional*):
                Original image size.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_views, num_channels, height, width)`:
                Pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.decoder_num_channels
        original_image_size = (
            original_image_size
            if original_image_size is not None
            else (self.config.image_size, self.config.image_size)
        )
        original_height, original_width = original_image_size
        num_patches_h = original_height // patch_size
        num_patches_w = original_width // patch_size
        # sanity check
        if num_patches_h * num_patches_w * self.num_views != patchified_pixel_values.shape[1]:
            raise ValueError(
                f"The number of patches in the patchified pixel values {patchified_pixel_values.shape[1]}, does not match the number of patches on original image {num_patches_h}*{num_patches_w}*{self.num_views}"
            )

        # unpatchify
        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size * self.num_views,
            num_patches_h,
            num_patches_w,
            patch_size,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            self.num_views,
            num_channels,
            num_patches_h * patch_size,
            num_patches_w * patch_size,
        )
        return pixel_values

    def forward_loss(self, pixel_values, pred, mask):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_views, num_channels, height, width)`):
                Pixel values.
            pred (`torch.FloatTensor` of shape `(batch_size, num_patches * num_views, patch_size**2 * num_channels)`:
                Predicted pixel values.
            mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Tensor indicating which patches are masked (1) and which are not (0).

        Returns:
            `torch.FloatTensor`: Pixel reconstruction loss.
        """
        target = self.patchify(pixel_values)
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def random_masking(self, sequence, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def forward(
        self,
        x: dict,
        return_recon: bool = False,
    ) -> Dict[str, torch.Tensor]:
        pixel_values = x['output_image']
        x = x['input_image']
        
        
        B, V, C, H, W = x.shape
        # shape: (batch * view, channels, img_height, img_width)
        x = x.reshape(B * V, C, H, W)
        
        # create patch embeddings and add position embeddings; remove CLS token
        embedding_output = self.vit.embeddings(
            x, bool_masked_pos=None, interpolate_pos_encoding=False,
        )[:, 1:]
        # shape: (batch * view, num_patches, embedding_dim)
        
        # # Add view embeddings to the sequence
        # # @Yanchne: I removed view embeddings because it gives more augmentations.
        # embedding_output = embedding_output.reshape(B, V, self.num_patches, self.hidden_size)
        # embedding_output += self.view_embeddings

        # reshape embedding_output to (batch, view * num_patches, embedding_dim)
        embedding_output = embedding_output.reshape(B, V * self.num_patches, self.hidden_size)

        # masking: length -> length * config.mask_ratio
        embeddings, mask, ids_restore = self.random_masking(embedding_output)

        # push data through vit encoder
        encoder_outputs = self.vit.encoder(
            embeddings,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=None,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.vit.layernorm(sequence_output)
        # shape: (batch, length, embedding_dim)
        # Decoder forward pass
        decoder_outputs = self.decoder(sequence_output, ids_restore)
        # shape: (batch, full length, embedding_dim)
        logits = decoder_outputs.logits
        
        loss = self.forward_loss(pixel_values, logits, mask)
        
        return_dict = {
            'logits': logits,
            'loss': loss,
            'mask': mask,
            'ids_restore': ids_restore,
            'reconstructions': self.unpatchify(logits) if return_recon else None,
        }
        return return_dict

    def get_model_outputs(self, batch_dict: dict, return_recon: bool = False) -> dict:
        x = batch_dict['input_image']
        
        B, V, C, H, W = x.shape
        # shape: (batch * view, channels, img_height, img_width)
        x = x.reshape(B * V, C, H, W)
        
        # create patch embeddings and add position embeddings; remove CLS token
        embedding_output = self.vit.embeddings(
            x, bool_masked_pos=None, interpolate_pos_encoding=False
        )[:, 1:]
        # shape: (batch * view, num_patches, embedding_dim)

        # reshape embedding_output to (batch, view * num_patches, embedding_dim)
        embedding_output = embedding_output.reshape(B, V * self.num_patches, self.hidden_size)

        # push data through vit encoder
        encoder_outputs = self.vit.encoder(
            embedding_output,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=None,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.vit.layernorm(sequence_output)

        if not return_recon:
            return {'latents': sequence_output}
        else:
            # no masking
            mask = torch.zeros_like(embedding_output)[..., 0]
            ids_restore = torch.arange(embedding_output.shape[1]).unsqueeze(0).repeat(B, 1).to(embedding_output.device)
            decoder_outputs = self.decoder(sequence_output, ids_restore)
            logits = decoder_outputs.logits
            reconstructions = self.unpatchify(logits)
            return {'logits': logits, 'reconstructions': reconstructions, 'mask': mask}

    def compute_loss(
        self,
        stage: str,
        **kwargs,
    ) -> tuple[torch.tensor, list[dict]]:
        assert 'loss' in kwargs, "Loss is not in the kwargs"
        mse_loss = kwargs['loss']
        loss = mse_loss
        return loss

    def predict_step(self, batch_dict: dict, return_recon: bool = False) -> dict:
        # get model outputs
        results_dict = self.get_model_outputs(batch_dict, return_recon=return_recon)
        return results_dict

class MultiViewTransformer(torch.nn.Module):
    """Vision Transformer implementation."""

    def __init__(self, config):
        super().__init__()
        # Set up ViT architecture
        self.mask_ratio = config['model']['model_params']['mask_ratio']
        self.avail_views = config['data']['avail_views']
        self.num_views = len(self.avail_views)
        config['model']['model_params']['num_views'] = self.num_views
        self.config = ViTMAEConfig(**config['model']['model_params'])
        MAE_VIT_SMALL_PATH = config['model']['pretrained']
        if 'vit-small-patch16-224.pth' in MAE_VIT_SMALL_PATH:
            config['model']['pretrained'] = 'facebook/dino-vits16'
        self.vit = ViTModel.from_pretrained(config['model']['pretrained'], use_safetensors=True)
        # fix sin-cos embedding
        self.vit.embeddings.position_embeddings.requires_grad = False
        
        h = w = config['model']['model_params']['image_size'] // config['model']['model_params']['patch_size']
        self.num_patches = h * w
        self.hidden_size = config['model']['model_params']['hidden_size']
        # self.view_embeddings = torch.nn.Parameter(torch.randn(1, self.num_views, self.num_patches, self.hidden_size))
        self.decoder = MultiViewTransformerDecoder(self.config, self.num_patches, self.num_views)
        if 'vit-small-patch16-224.pth' in MAE_VIT_SMALL_PATH:
        #   self.vit, self.decoder = load_mae_ckpt(MAE_VIT_SMALL_PATH, self.vit, self.decoder)
          self.vit, _ = load_mae_ckpt(MAE_VIT_SMALL_PATH, self.vit, None)

    def patchify(self, pixel_values):
        patch_size, num_channels = self.config.patch_size, self.config.decoder_num_channels
        # sanity checks
        if pixel_values.shape[2] != num_channels:
            raise ValueError(
                "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
            )

        # patchify
        batch_size = pixel_values.shape[0]
        num_views = pixel_values.shape[1]
        num_patches_h = pixel_values.shape[3] // patch_size
        num_patches_w = pixel_values.shape[4] // patch_size
        patchified_pixel_values = pixel_values.reshape(
            batch_size*num_views, num_channels, num_patches_h, patch_size, num_patches_w, patch_size
        )
        patchified_pixel_values = torch.einsum("nchpwq->nhwpqc", patchified_pixel_values)
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size, num_patches_h * num_patches_w * num_views, patch_size**2 * num_channels
        )
        return patchified_pixel_values

    def unpatchify(self, patchified_pixel_values, original_image_size: Optional[tuple[int, int]] = None):
        """
        Args:
            patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
            original_image_size (`tuple[int, int]`, *optional*):
                Original image size.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_views, num_channels, height, width)`:
                Pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.decoder_num_channels
        original_image_size = (
            original_image_size
            if original_image_size is not None
            else (self.config.image_size, self.config.image_size)
        )
        original_height, original_width = original_image_size
        num_patches_h = original_height // patch_size
        num_patches_w = original_width // patch_size
        # sanity check
        if num_patches_h * num_patches_w * self.num_views != patchified_pixel_values.shape[1]:
            raise ValueError(
                f"The number of patches in the patchified pixel values {patchified_pixel_values.shape[1]}, does not match the number of patches on original image {num_patches_h}*{num_patches_w}*{self.num_views}"
            )

        # unpatchify
        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size * self.num_views,
            num_patches_h,
            num_patches_w,
            patch_size,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            self.num_views,
            num_channels,
            num_patches_h * patch_size,
            num_patches_w * patch_size,
        )
        return pixel_values

    def forward_loss(self, pixel_values, pred, mask):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_views, num_channels, height, width)`):
                Pixel values.
            pred (`torch.FloatTensor` of shape `(batch_size, num_patches * num_views, patch_size**2 * num_channels)`:
                Predicted pixel values.
            mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Tensor indicating which patches are masked (1) and which are not (0).

        Returns:
            `torch.FloatTensor`: Pixel reconstruction loss.
        """
        target = self.patchify(pixel_values)
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def random_masking(self, sequence, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def adjust_mask_and_ids_restore_after_reshape(
        self, 
        mask: torch.Tensor, 
        ids_restore: torch.Tensor, 
        batch_size: int, 
        num_views: int, 
        num_patches: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Adjust mask and ids_restore after reshaping sequence_output from (B*V, ...) to (B, ...).
        
        Args:
            mask: Tensor of shape (B * V, num_patches) indicating which patches are masked
            ids_restore: Tensor of shape (B * V, num_patches) with restore indices per view
            batch_size: Batch size B
            num_views: Number of views V
            num_patches: Number of patches per view
            
        Returns:
            new_mask: Tensor of shape (B, V * num_patches) - mask concatenated across views
            new_ids_restore: Tensor of shape (B, V * num_patches) - ids_restore with proper offsets
        """
        device = mask.device
        
        # Reshape mask from (B * V, num_patches) to (B, V * num_patches)
        new_mask = mask.reshape(batch_size, num_views * num_patches)
        
        # Adjust ids_restore: each view's indices need to be offset
        # View 0: 0 to num_patches-1
        # View 1: num_patches to 2*num_patches-1
        # View 2: 2*num_patches to 3*num_patches-1, etc.
        new_ids_restore = ids_restore.reshape(batch_size, num_views, num_patches)
        # Create offset tensor: [0, num_patches, 2*num_patches, ...]
        offsets = torch.arange(num_views, device=device) * num_patches
        offsets = offsets.view(1, num_views, 1)  # (1, V, 1)
        
        # Add offsets to each view's restore indices
        new_ids_restore = new_ids_restore + offsets
        # Reshape to (B, V * num_patches)
        new_ids_restore = new_ids_restore.reshape(batch_size, num_views * num_patches)

        return new_mask, new_ids_restore

    def forward(
        self,
        x: dict,
        return_recon: bool = False,
    ) -> Dict[str, torch.Tensor]:
        pixel_values = x['output_image']
        x = x['input_image']
        
        
        B, V, C, H, W = x.shape
        # shape: (batch * view, channels, img_height, img_width)
        x = x.reshape(B * V, C, H, W)
        
        # create patch embeddings and add position embeddings; remove CLS token
        embedding_output = self.vit.embeddings(
            x, bool_masked_pos=None, interpolate_pos_encoding=False,
        )[:, 1:]
        # shape: (batch * view, num_patches, embedding_dim)
        
        # # Add view embeddings to the sequence
        # # @Yanchne: I removed view embeddings because it gives more augmentations.
        # embedding_output = embedding_output.reshape(B, V, self.num_patches, self.hidden_size)
        # embedding_output += self.view_embeddings

        # reshape embedding_output to (batch, view * num_patches, embedding_dim)
        # embedding_output = embedding_output.reshape(B, V * self.num_patches, self.hidden_size)

        # masking: length -> length * config.mask_ratio
        embeddings, mask, ids_restore = self.random_masking(embedding_output)

        # push data through vit encoder
        encoder_outputs = self.vit.encoder(
            embeddings,
            # head_mask=None,
            # output_attentions=False,
            # output_hidden_states=False,
            # return_dict=None,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.vit.layernorm(sequence_output)

        # Reshape sequence_output to (batch, view * num_patches * (1 - mask_ratio), embedding_dim)
        sequence_output = sequence_output.reshape(B, -1, self.hidden_size)
        mask, ids_restore = self.adjust_mask_and_ids_restore_after_reshape(mask, ids_restore, B, V, self.num_patches)
        
        # shape: (batch, length, embedding_dim)
        # Decoder forward pass
        decoder_outputs = self.decoder(sequence_output, ids_restore)
        # shape: (batch, full length, embedding_dim)
        logits = decoder_outputs.logits
        
        loss = self.forward_loss(pixel_values, logits, mask)
        
        return_dict = {
            'logits': logits,
            'loss': loss,
            'mask': mask,
            'ids_restore': ids_restore,
            'reconstructions': self.unpatchify(logits) if return_recon else None,
        }
        return return_dict

    def get_model_outputs(self, batch_dict: dict, return_recon: bool = False) -> dict:
        x = batch_dict['input_image']
        
        B, V, C, H, W = x.shape
        # shape: (batch * view, channels, img_height, img_width)
        x = x.reshape(B * V, C, H, W)
        
        # create patch embeddings and add position embeddings; remove CLS token
        embedding_output = self.vit.embeddings(
            x, bool_masked_pos=None, interpolate_pos_encoding=False
        )[:, 1:]
        # shape: (batch * view, num_patches, embedding_dim)

        # reshape embedding_output to (batch, view * num_patches, embedding_dim)
        embedding_output = embedding_output.reshape(B, V * self.num_patches, self.hidden_size)

        # push data through vit encoder
        encoder_outputs = self.vit.encoder(
            embedding_output,
            # head_mask=None,
            # output_attentions=False,
            # output_hidden_states=False,
            # return_dict=None,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.vit.layernorm(sequence_output)

        if not return_recon:
            return {'latents': sequence_output}
        else:
            # no masking
            mask = torch.zeros_like(embedding_output)[..., 0]
            ids_restore = torch.arange(embedding_output.shape[1]).unsqueeze(0).repeat(B, 1).to(embedding_output.device)
            decoder_outputs = self.decoder(sequence_output, ids_restore)
            logits = decoder_outputs.logits
            reconstructions = self.unpatchify(logits)
            return {'logits': logits, 'reconstructions': reconstructions, 'mask': mask}

    def compute_loss(
        self,
        stage: str,
        **kwargs,
    ) -> tuple[torch.tensor, list[dict]]:
        assert 'loss' in kwargs, "Loss is not in the kwargs"
        mse_loss = kwargs['loss']
        loss = mse_loss
        return loss

    def predict_step(self, batch_dict: dict, return_recon: bool = False) -> dict:
        # get model outputs
        results_dict = self.get_model_outputs(batch_dict, return_recon=return_recon)
        return results_dict

class MultiViewTransformerDecoder(ViTMAEDecoder):
    def __init__(self, config, num_patches, num_views):
        super().__init__(config, num_patches) # 196 is the number of patches in the decoder

        # re-initialize the decoder_pred based on the decoder_num_channels
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.patch_size**2 * config.decoder_num_channels, bias=True
        )  # encoder to decoder

        # # re-initialize the decoder position embeddings per view
        # decoder_pos_embed_dict = {}
        # for i in range(num_views):
        #     decoder_pos_embed_dict[i] = torch.from_numpy(
        #         get_2d_sincos_pos_embed(config.decoder_hidden_size, int(num_patches**0.5), add_cls_token=False)
        #     ).float().unsqueeze(0)
        # self.decoder_pos_embed = nn.Parameter(torch.cat([decoder_pos_embed_dict[i] for i in range(num_views)], dim=1), requires_grad=False)
        
        # learnable view embeddings
        self.decoder_view_embed = torch.nn.Parameter(
            torch.randn(1, num_views*num_patches, config.decoder_hidden_size), requires_grad=True)

    def forward(self, hidden_states, ids_restore):
        # embed tokens
        x = self.decoder_embed(hidden_states)
        B, _, D = x.shape

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1) 
        # unshuffle
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]).to(x_.device))
        # add pos embed
        x = x.reshape(B * self.config.num_views, -1, D)
        hidden_states = x + self.decoder_pos_embed[:,1:] # skip cls token
        hidden_states = hidden_states.reshape(B, -1, D)
        
        # # add view embeddings
        hidden_states += self.decoder_view_embed

        # apply Transformer layers (blocks)
        for i, layer_module in enumerate(self.decoder_layers):
            hidden_states = layer_module(hidden_states)

        hidden_states = self.decoder_norm(hidden_states)

        # predictor projection
        logits = self.decoder_pred(hidden_states)

        return ViTMAEDecoderOutput(
            logits=logits,
        ) 

class MultiViewTransformerDecoder_OLD(ViTMAEDecoder):
    def __init__(self, config, num_patches, num_views):
        super().__init__(config, num_patches) # 196 is the number of patches in the decoder

        # re-initialize the decoder_pred based on the decoder_num_channels
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.patch_size**2 * config.decoder_num_channels, bias=True
        )  # encoder to decoder

        # # re-initialize the decoder position embeddings per view
        # decoder_pos_embed_dict = {}
        # for i in range(num_views):
        #     decoder_pos_embed_dict[i] = torch.from_numpy(
        #         get_2d_sincos_pos_embed(config.decoder_hidden_size, int(num_patches**0.5), add_cls_token=False)
        #     ).float().unsqueeze(0)
        # self.decoder_pos_embed = nn.Parameter(torch.cat([decoder_pos_embed_dict[i] for i in range(num_views)], dim=1), requires_grad=False)
        
        # learnable view embeddings
        self.decoder_view_embed = torch.nn.Parameter(
            torch.randn(1, num_views*num_patches, config.decoder_hidden_size), requires_grad=True)

    def forward(self, hidden_states, ids_restore):
        # embed tokens
        x = self.decoder_embed(hidden_states)
        B, _, D = x.shape

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1) 
        # unshuffle
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]).to(x_.device))
        # add pos embed
        x = x.reshape(B * self.config.num_views, -1, D)
        hidden_states = x + self.decoder_pos_embed[:,1:] # skip cls token
        hidden_states = hidden_states.reshape(B, -1, D)
        
        # # add view embeddings
        hidden_states += self.decoder_view_embed

        # apply Transformer layers (blocks)
        for i, layer_module in enumerate(self.decoder_layers):

            layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=False)

            hidden_states = layer_outputs[0]

        hidden_states = self.decoder_norm(hidden_states)

        # predictor projection
        logits = self.decoder_pred(hidden_states)

        return ViTMAEDecoderOutput(
            logits=logits,
        ) 


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Create 2D sin/cos positional embeddings.

    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.

    Returns:
        (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the
        position embeddings (with or without classification token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class ViTMAE(ViTMAEForPreTraining):

    # Overriding the forward method to return the latent and loss
    # This is used for training and inference
    # Huggingface Transformer library
    def forward(
        self,
        pixel_values: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_latent: bool = False,
        return_recon: bool = False,
        output_views: Optional[torch.Tensor] = None,
        output_images: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Setting default for return_dict based on the configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if (self.training or self.config.mask_ratio > 0) or return_recon:
            outputs = self.vit(
                pixel_values,
                noise=noise,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            latent = outputs.last_hidden_state
        else:
            # use for fine-tuning, or inference
            # mask_ratio = 0
            embedding_output, mask, ids_restore = self.vit.embeddings(pixel_values)
            embedding_output_ = embedding_output[:, 1:, :]  # no cls token
            # unshuffle the embedding output
            index = ids_restore.unsqueeze(-1).repeat(
                1, 1, embedding_output_.shape[2]
            ).to(embedding_output_.device)
            embedding_output_ = torch.gather(embedding_output_, dim=1, index=index)
            # add cls token back
            embedding_output = torch.cat((embedding_output[:, :1, :], embedding_output_), dim=1)
            encoder_outputs = self.vit.encoder(
                embedding_output,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
            latent = self.vit.layernorm(sequence_output)
            if not return_latent:
                # return the cls token and 0 loss if not return_latent
                return latent[:, 0], 0
        if return_latent:
            return latent
        # extract cls latent
        cls_latent = latent[:, 0]  # shape (batch_size, hidden_size)
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs = self.decoder(latent, ids_restore, output_views)
        logits = decoder_outputs.logits
        # shape (batch_size, num_patches, patch_size*patch_size*num_channels)
        if output_images is not None:
            # the mask should be all 1s
            mask = torch.ones_like(mask)
            loss = self.forward_loss(output_images, logits, mask)
        else:
            loss = self.forward_loss(pixel_values, logits, mask)
        if return_recon:
            return {
                'latents': latent,
                'loss': loss,
                'reconstructions': self.unpatchify(logits),
            }
        return {
            'latents': cls_latent,
            'loss': loss,
            'logits': logits,
        }

def main():
    decoder = MultiViewDecoder(num_views=2)
    # model = VisionTransformer()

if __name__ == "__main__":
    main()
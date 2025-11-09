from typing import Dict, Optional

import torch
import torch.nn as nn
import numpy as np
from jaxtyping import Float
from transformers import ViTMAEConfig, ViTMAEForPreTraining, ViTModel
from transformers.models.vit_mae.modeling_vit_mae import (ViTMAEDecoder,
                                                          ViTMAEDecoderOutput)

class MultiViewTransformer(torch.nn.Module):
    """Vision Transformer implementation."""

    def __init__(self, config):
        super().__init__()
        # Set up ViT architecture
        self.config = ViTMAEConfig(**config['model']['model_params'])
        self.mask_ratio = config['model']['model_params']['mask_ratio']
        self.avail_views = config['data']['avail_views']
        self.num_views = len(self.avail_views)
        self.vit = ViTModel.from_pretrained(config['model']['pretrained'], use_safetensors=True)

        # fix sin-cos embedding
        self.vit.embeddings.position_embeddings.requires_grad = False
        
        h = w = config['model']['model_params']['image_size'] // config['model']['model_params']['patch_size']
        self.num_patches = h * w
        self.hidden_size = config['model']['model_params']['hidden_size']
        self.view_embeddings = torch.nn.Parameter(torch.randn(1, self.num_views, self.num_patches, self.hidden_size))

        self.decoder = MultiViewTransformerDecoder(self.config, self.num_patches, self.num_views)

    def patchify(self, pixel_values):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_views, num_channels, height, width)`):
                Pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_patches * num_views, patch_size**2 * num_channels)`:
                Patchified pixel values.
        """
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
            patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_views, num_patches, patch_size**2 * num_channels)`:
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
        num_views = patchified_pixel_values.shape[1]
        num_patches_h = original_height // patch_size
        num_patches_w = original_width // patch_size
        # sanity check
        if num_patches_h * num_patches_w != patchified_pixel_values.shape[1]:
            raise ValueError(
                f"The number of patches in the patchified pixel values {patchified_pixel_values.shape[1]}, does not match the number of patches on original image {num_patches_h}*{num_patches_w}"
            )

        # unpatchify
        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size * num_views,
            num_patches_h,
            num_patches_w,
            patch_size,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_views,
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
        
        # Add view embeddings to the sequence
        embedding_output = embedding_output.reshape(B, V, self.num_patches, self.hidden_size)
        embedding_output += self.view_embeddings

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
        }
        return return_dict

    def get_model_outputs(self, batch_dict: dict, return_images: bool = True) -> dict:
        results_dict = self.forward(batch_dict)
        if return_images:
            results_dict['images'] = batch_dict['image']
        return results_dict

    def compute_loss(
        self,
        stage: str,
        **kwargs,
    ) -> tuple[torch.tensor, list[dict]]:
        assert 'loss' in kwargs, "Loss is not in the kwargs"
        mse_loss = kwargs['loss']
        loss = mse_loss
        return loss

    def predict_step(self, batch_dict: dict) -> dict:
        # set mask_ratio to 0 for inference
        self.config.mask_ratio = 0
        # get model outputs
        results_dict = self.get_model_outputs(batch_dict, return_images=False)
        # reset mask_ratio to the original value
        self.vit_mae.config.mask_ratio = self.mask_ratio
        return results_dict

class MultiViewTransformerDecoder(ViTMAEDecoder):
    def __init__(self, config, num_patches, num_views):
        super().__init__(config, num_patches) # 196 is the number of patches in the decoder

        # re-initialize the decoder_pred based on the decoder_num_channels
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.patch_size**2 * config.decoder_num_channels, bias=True
        )  # encoder to decoder

        # re-initialize the decoder position embeddings per view
        decoder_pos_embed_dict = {}
        for i in range(num_views):
            decoder_pos_embed_dict[i] = torch.from_numpy(
                get_2d_sincos_pos_embed(config.decoder_hidden_size, int(num_patches**0.5), add_cls_token=False)
            ).float().unsqueeze(0)
        self.decoder_pos_embed = nn.Parameter(torch.cat([decoder_pos_embed_dict[i] for i in range(num_views)], dim=1), requires_grad=False)
        
        # learnable view embeddings
        self.decoder_view_embed = torch.nn.Parameter(
            torch.randn(1, num_views*num_patches, config.decoder_hidden_size), requires_grad=True)

    def forward(self, hidden_states, ids_restore):
        # embed tokens
        x = self.decoder_embed(hidden_states)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1) 
        # unshuffle
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]).to(x_.device))
        # add pos embed
        decoder_pos_embed = self.decoder_pos_embed
        hidden_states = x + decoder_pos_embed
        # add view embeddings
        decoder_view_embed_ = self.decoder_view_embed
        hidden_states = hidden_states + decoder_view_embed_

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
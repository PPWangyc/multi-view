from typing import Dict, Optional

import torch
from jaxtyping import Float
from transformers import ViTMAEConfig, ViTMAEForPreTraining, ViTModel
from transformers.models.vit_mae.modeling_vit_mae import (ViTMAEDecoder,
                                                          ViTMAEDecoderOutput)


class MultiViewTransformer(torch.nn.Module):
    """Vision Transformer implementation."""

    def __init__(self, config):
        super().__init__()
        # Set up ViT architecture
        # self.vit_mae_config = ViTMAEConfig(**config['model']['model_params'])
        # self.vit_mae = ViTMAE(self.vit_mae_config).from_pretrained("facebook/vit-mae-base")
        self.mask_ratio = config['model']['model_params']['mask_ratio']
        self.num_views = config['model']['model_params']['num_views']
        self.avail_views = config['data']['avail_views']
        self.vit = ViTModel.from_pretrained("facebook/dino-vitb16")
        
        self.view2idx = {view: idx for idx, view in enumerate(self.avail_views)}
        self.idx2view = {idx: view for view, idx in self.view2idx.items()}
        h = w = config['model']['model_params']['image_size'] // config['model']['model_params']['patch_size']
        self.num_patches = h * w
        self.hidden_size = config['model']['model_params']['hidden_size']
        self.view_embeddings = torch.nn.Parameter(torch.randn(self.num_views, self.num_patches, self.hidden_size))

    def forward(
        self,
        x: dict,
    ) -> Dict[str, torch.Tensor]:
        x = x['image']
        # create patch embeddings and add position embeddings; remove CLS token
        embedding_output = self.vit.embeddings(
            x, bool_masked_pos=None, interpolate_pos_encoding=False,
        )[:, 1:]
        # shape: (view * batch, num_patches, embedding_dim)
        
        # TODO: Add view embeddings to the sequence
        # @Yanchen: 11/06/2025

        # concat embeddings output across views
        # (view * batch, num_patches, embedding_dim) -> (batch, view * num_patches, embedding_dim)
        embedding_output = embedding_output.reshape(-1, self.num_views * self.num_patches, self.hidden_size)

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
        # shape: (view * batch, num_patches, embedding_dim)
        print(f'shape of sequence_output: {sequence_output.shape}')
        print(f'shape of mask: {mask.shape}')
        print(f'shape of ids_restore: {ids_restore.shape}')
        # TODO: Add Decoder for reconstruction
        # @Yanchen: 11/06/2025
        exit()
        return sequence_output

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
        self.vit_mae.config.mask_ratio = 0
        # get model outputs
        results_dict = self.get_model_outputs(batch_dict, return_images=False)
        # reset mask_ratio to the original value
        self.vit_mae.config.mask_ratio = self.mask_ratio
        return results_dict


# class MultiViewTransformer(torch.nn.Module):
#     """Vision Transformer implementation."""

#     def __init__(self, config):
#         super().__init__()
#         # Set up ViT architecture
#         self.vit_mae_config = ViTMAEConfig(**config['model']['model_params'])
#         self.vit_mae = ViTMAE(self.vit_mae_config).from_pretrained("facebook/vit-mae-base")
#         self.mask_ratio = config['model']['model_params']['mask_ratio']

#     def forward(
#         self,
#         x: dict,
#     ) -> Dict[str, torch.Tensor]:
#         x = x['image']
#         results_dict = self.vit_mae(pixel_values=x, return_recon=True)
#         return results_dict

#     def get_model_outputs(self, batch_dict: dict, return_images: bool = True) -> dict:
#         results_dict = self.forward(batch_dict)
#         if return_images:
#             results_dict['images'] = batch_dict['image']
#         return results_dict

#     def compute_loss(
#         self,
#         stage: str,
#         **kwargs,
#     ) -> tuple[torch.tensor, list[dict]]:
#         assert 'loss' in kwargs, "Loss is not in the kwargs"
#         mse_loss = kwargs['loss']
#         loss = mse_loss
#         return loss

#     def predict_step(self, batch_dict: dict) -> dict:
#         # set mask_ratio to 0 for inference
#         self.vit_mae.config.mask_ratio = 0
#         # get model outputs
#         results_dict = self.get_model_outputs(batch_dict, return_images=False)
#         # reset mask_ratio to the original value
#         self.vit_mae.config.mask_ratio = self.mask_ratio
#         return results_dict

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
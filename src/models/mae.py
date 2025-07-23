from transformers import ViTMAEConfig, ViTMAEForPreTraining
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEDecoder, ViTMAEDecoderOutput
import torch
from typing import Dict, Optional
from jaxtyping import Float

class VisionTransformer(torch.nn.Module):
    """Vision Transformer implementation."""

    def __init__(self, config):
        super().__init__()
        # Set up ViT architecture
        self.vit_mae_config = ViTMAEConfig(**config['model']['model_params'])
        self.vit_mae = ViTMAE(self.vit_mae_config).from_pretrained("facebook/vit-mae-base")
        self.mask_ratio = config['model']['model_params']['mask_ratio']

    def forward(
        self,
        x: Float[torch.Tensor, 'batch channels img_height img_width'],
    ) -> Dict[str, torch.Tensor]:
        x = x['image']
        results_dict = self.vit_mae(pixel_values=x, return_recon=True)
        return results_dict

    def get_model_outputs(self, batch_dict: dict, return_images: bool = True) -> dict:
        x = batch_dict['image']
        results_dict = self.forward(x)
        if return_images:
            results_dict['images'] = x
        return results_dict

    def compute_loss(
        self,
        stage: str,
        **kwargs,
    ) -> tuple[torch.tensor, list[dict]]:
        assert 'loss' in kwargs, "Loss is not in the kwargs"
        mse_loss = kwargs['loss']
        # add all losses here for logging
        log_list = [
            {'name': f'{stage}_mse', 'value': mse_loss.clone()}
        ]
        loss = mse_loss
        return loss

    def predict_step(self, batch_dict: dict, batch_idx: int) -> dict:
        # set mask_ratio to 0 for inference
        self.vit_mae.config.mask_ratio = 0
        # get model outputs
        results_dict = self.get_model_outputs(batch_dict, return_images=False)
        # reset mask_ratio to the original value
        self.vit_mae.config.mask_ratio = self.mask_ratio
        results_dict['metadata'] = {
            'video': batch_dict['video'],
            'idx': batch_dict['idx'],
            'image_paths': batch_dict['image_path'],
        }
        return results_dict

class MVVisionTransformer(VisionTransformer):
    def __init__(self, config):
        super().__init__(config)
        avail_views = config['data']['avail_views']
        self.view2idx = {view: idx for idx, view in enumerate(avail_views)}
        assert self.vit_mae_config.num_views > 1, "Multi-view ViT should have more than 1 view"
        self.vit_mae.decoder = MultiViewDecoder(self.vit_mae_config)
        # Add another embed in the decoder
        self.vit_mae.from_pretrained("facebook/vit-mae-base")
        self.mask_ratio = config['model']['model_params']['mask_ratio']

    def forward(
        self,
        batch_dict: dict,
    ) -> Dict[str, torch.Tensor]:
        input_images = batch_dict['input_image']
        output_images = batch_dict['output_image']
        output_views = batch_dict['output_view']
        output_views = torch.tensor([self.view2idx[view] for view in output_views])
        results_dict = self.vit_mae(
            pixel_values=input_images, 
            output_views=output_views, 
            output_images=output_images, 
            return_recon=True
        )
        return results_dict


class MultiViewDecoder(ViTMAEDecoder):
    def __init__(self, config):
        super().__init__(config, 196) # 196 is the number of patches in the decoder
        # output_view embeddings
        self.decoder_view_embed = torch.nn.Parameter(torch.randn(config.num_views, 197, config.decoder_hidden_size))

    def get_decoder_view_embed(self, output_views: torch.Tensor) -> torch.Tensor:
        """
        Extract decoder view embeddings based on output_views.

        Args:
            output_views: Tensor of shape [batch_size] containing view indices
        Returns:
            decoder_view_embed_: Tensor of shape [batch_size, 197, decoder_hidden_size]
        """
        # Index into decoder_view_embed using output_views
        return self.decoder_view_embed[output_views]

    def forward(self, hidden_states, ids_restore, output_views):
        # embed tokens
        x = self.decoder_embed(hidden_states)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # unshuffle
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]).to(x_.device))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # add pos embed
        decoder_pos_embed = self.decoder_pos_embed
        hidden_states = x + decoder_pos_embed
        # w
        # add view embeddings
        decoder_view_embed_ = self.get_decoder_view_embed(output_views)
        hidden_states = hidden_states + decoder_view_embed_

        # apply Transformer layers (blocks)
        for i, layer_module in enumerate(self.decoder_layers):

            layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=False)

            hidden_states = layer_outputs[0]


        hidden_states = self.decoder_norm(hidden_states)

        # predictor projection
        logits = self.decoder_pred(hidden_states)  

        # remove cls token
        logits = logits[:, 1:, :]

        return ViTMAEDecoderOutput(
            logits=logits,
        ) 

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
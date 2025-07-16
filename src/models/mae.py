from beast.models.vits import (
    ViTMAE,
)
from transformers import ViTMAEConfig
import torch
from typing import Dict
from jaxtyping import Float


class VisionTransformer():
    """Vision Transformer implementation."""

    def __init__(self):
        super().__init__()
        # Set up ViT architecture
        vit_mae_config = ViTMAEConfig()
        self.vit_mae = ViTMAE(vit_mae_config).from_pretrained("facebook/vit-mae-base")
        self.mask_ratio = 0.75

    def forward(
        self,
        x: Float[torch.Tensor, 'batch channels img_height img_width'],
    ) -> Dict[str, torch.Tensor]:
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

def main():
    model = VisionTransformer()
    print(model)

if __name__ == "__main__":
    main()
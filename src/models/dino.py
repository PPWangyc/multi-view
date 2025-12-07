from typing import Dict, Optional

import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel


class VisionTransformer(torch.nn.Module):
    """Vision Transformer implementation using DINOv3."""

    def __init__(self, config):
        super().__init__()
        # Set up DINOv3 architecture
        model_params = config['model']['model_params']
        # Get pretrained model name if specified, otherwise use default
        pretrained_name = config['model'].get('pretrained', 'facebook/dinov3-base')
        
        # Initialize config with model params
        self.dinov3_config = DINOv3ViTConfig(**model_params)
        
        # Load pretrained model
        self.dinov3 = DINOv3ViTModel.from_pretrained(pretrained_name)
        
        # Update config if needed (in case pretrained model has different config)
        if hasattr(self.dinov3.config, 'image_size'):
            self.dinov3_config = self.dinov3.config

    def forward(
        self,
        x: dict,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through DINOv3 model.
        
        Parameters
        ----------
        x: dict
            Dictionary containing 'image' key with images of shape (batch, 3, H, W)
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing 'latents' (CLS token), 'last_hidden_state', and 'loss'
            Note: 'loss' is a placeholder (zero tensor). For pretraining, implement your own loss.
        """
        pixel_values = x['image']
        outputs = self.dinov3(pixel_values=pixel_values, return_dict=True)
        
        # Extract CLS token (first token) as latent representation
        cls_latent = outputs.last_hidden_state[:, 0]  # shape (batch_size, hidden_size)
        
        # Placeholder loss - computed from latents to allow gradients to flow
        # NOTE: This is a dummy loss for compatibility. For actual DINOv3 pretraining,
        # you should implement self-distillation loss (student-teacher with EMA).
        # DINOv3 doesn't have a built-in loss like MAE.
        # This minimal loss (tiny L2 regularization) ensures gradients can flow through the model.
        placeholder_loss = (cls_latent ** 2).mean() * 1e-8  # Tiny loss to allow gradients
        
        results_dict = {
            'latents': cls_latent,
            'last_hidden_state': outputs.last_hidden_state,
            'loss': placeholder_loss,
        }
        return results_dict

    def get_model_outputs(self, batch_dict: dict, return_images: bool = True) -> dict:
        """Get model outputs for logging/visualization.
        
        Parameters
        ----------
        batch_dict: dict
            Input batch dictionary
        return_images: bool
            Whether to include images in output
        
        Returns
        -------
        dict
            Dictionary containing model outputs
        """
        results_dict = self.forward(batch_dict)
        if return_images:
            results_dict['images'] = batch_dict['image']
        return results_dict

    def compute_loss(
        self,
        stage: str,
        **kwargs,
    ) -> tuple[torch.tensor, list[dict]]:
        """Compute loss for training.
        
        Note: DINOv3 is typically pretrained using self-distillation.
        For custom pretraining, you may need to implement your own loss function.
        This is a placeholder that expects 'loss' in kwargs.
        
        Parameters
        ----------
        stage: str
            Training stage ('train', 'val', etc.)
        **kwargs
            Must contain 'loss' key
        
        Returns
        -------
        torch.tensor
            Loss tensor
        """
        assert 'loss' in kwargs, "Loss is not in the kwargs"
        loss = kwargs['loss']
        # add all losses here for logging
        log_list = [
            {'name': f'{stage}_loss', 'value': loss.clone()}
        ]
        return loss

    def predict_step(self, batch_dict: dict) -> dict:
        """Prediction step for inference.
        
        Parameters
        ----------
        batch_dict: dict
            Input batch dictionary
        
        Returns
        -------
        dict
            Model outputs without images
        """
        # get model outputs
        results_dict = self.get_model_outputs(batch_dict, return_images=False)
        return results_dict


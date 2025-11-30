from typing import Dict

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, VideoMAEForPreTraining


class VideoMAE(torch.nn.Module):
    """VideoMAE implementation for video pretraining."""

    def __init__(self, config):
        super().__init__()
        # Load pretrained VideoMAE model
        self.videomae = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")
        self.image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        
        # Calculate sequence length for masking
        self.mask_ratio = 0.9
        self.num_frames = 16  # Fixed for our dataset
        self.num_patches_per_frame = (self.videomae.config.image_size // self.videomae.config.patch_size) ** 2
        self.seq_length = (self.num_frames // self.videomae.config.tubelet_size) * self.num_patches_per_frame

    def _denormalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize ImageNet-normalized tensor back to [0, 1] range.
        
        Parameters
        ----------
        tensor: torch.Tensor
            Normalized tensor with ImageNet stats, shape (..., 3, H, W)
        
        Returns
        -------
        torch.Tensor
            Denormalized tensor in [0, 1] range
        """
        # ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # Reshape mean/std to broadcast correctly: (1, 1, 3, 1, 1) for (batch, frames, 3, H, W)
        # or (1, 3, 1, 1) for (frames, 3, H, W)
        if tensor.dim() == 4:  # (frames, 3, H, W)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
        else:  # (batch, frames, 3, H, W) or other shapes
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1).to(tensor.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1).to(tensor.device)
        return tensor * std + mean

    def _generate_masked_positions(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate boolean mask tensor with exactly mask_ratio * seq_length positions masked.
        
        Parameters
        ----------
        batch_size: int
            Batch size
        device: torch.device
            Device to create tensor on
        
        Returns
        -------
        torch.Tensor
            Boolean tensor of shape (batch_size, seq_length) where True indicates masked positions
        """
        num_masked = int(self.mask_ratio * self.seq_length)
        
        # Create boolean tensor initialized to False
        bool_masked_pos = torch.zeros((batch_size, self.seq_length), dtype=torch.bool, device=device)
        
        # For each sample in the batch, randomly select num_masked positions to mask
        for b in range(batch_size):
            # Randomly select indices to mask
            masked_indices = torch.randperm(self.seq_length, device=device)[:num_masked]
            bool_masked_pos[b, masked_indices] = True
        
        return bool_masked_pos

    def forward(
        self,
        x: dict,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through VideoMAE model.
        
        Parameters
        ----------
        x: dict
            Dictionary containing 'image' key with video clips of shape (batch, 16, 3, 224, 224)
            Images are expected to be ImageNet-normalized
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing 'loss' and optionally 'reconstructions'
        """
        video_clips = x['image']  # shape: (batch, 16, 3, 224, 224)
        batch_size = video_clips.shape[0]
        device = video_clips.device
        
        # # Denormalize from ImageNet stats to [0, 1] range
        # # video_clips is (batch, 16, 3, 224, 224), need to denormalize per frame
        # denormalized_clips = []
        # for b in range(batch_size):
        #     clip = video_clips[b]  # shape: (16, 3, 224, 224)
        #     # Denormalize each frame (clip is 4D: frames, channels, height, width)
        #     clip_denorm = self._denormalize_tensor(clip)  # (16, 3, 224, 224)
        #     # Clamp to [0, 1] and convert to numpy for processor
        #     clip_denorm = torch.clamp(clip_denorm, 0, 1)
        #     # Convert to numpy and then to PIL Images
        #     clip_np = clip_denorm.permute(0, 2, 3, 1).cpu().numpy()  # (16, 224, 224, 3)
        #     clip_np = (clip_np * 255).astype(np.uint8)
        #     # Convert to list of PIL Images
        #     frames = [Image.fromarray(clip_np[i]) for i in range(clip_np.shape[0])]
        #     denormalized_clips.append(frames)
        
        # # Process videos with image processor
        # # The processor expects a list of PIL Images per video
        # pixel_values_list = []
        # for video_frames in denormalized_clips:
        #     processed = self.image_processor(video_frames, return_tensors="pt")
        #     pixel_values_list.append(processed.pixel_values)
        
        # Stack all processed videos and move to device
        # pixel_values = torch.cat(pixel_values_list, dim=0).to(device)  # shape: (batch, num_frames, 3, 224, 224)
        
        # Generate masking positions with exact mask_ratio
        # bool_masked_pos shape: (batch, seq_length)
        bool_masked_pos = self._generate_masked_positions(batch_size, device)
        # Forward through model
        outputs = self.videomae(pixel_values=video_clips, bool_masked_pos=bool_masked_pos)
        results_dict = {
            'loss': outputs.loss,
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
    ) -> torch.tensor:
        """Compute loss for training.
        
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
        mse_loss = kwargs['loss']
        # add all losses here for logging
        log_list = [
            {'name': f'{stage}_mse', 'value': mse_loss.clone()}
        ]
        loss = mse_loss
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
        # For inference, we might want to set mask_ratio to 0 or use different masking
        # For now, use the same forward pass
        results_dict = self.get_model_outputs(batch_dict, return_images=False)
        return results_dict
"""
Script to generate pseudo 3D information from multi-view images using VGGT (Visual Geometry Grounded Transformer).

This script:
1. Loads multi-view images using MVTDataset
2. Runs VGGT inference to extract 3D information (depth maps, point clouds, camera parameters)
3. Saves pseudo 3D data for SSL training

VGGT (CVPR 2025 Best Paper) is a feed-forward neural network that can directly infer:
- Camera parameters
- Point maps
- Depth maps
- 3D point tracks
from one or multiple views.

Author: Generated for multi-view SSL training
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

# Add vggt directory to Python path
vggt_path = Path(__file__).parent.parent.parent / 'vggt'
if str(vggt_path) not in sys.path:
    sys.path.insert(0, str(vggt_path))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

from data.datasets import MVTDataset
from utils.log_utils import get_logger

logger = get_logger()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_vggt_model(model_name: Optional[str] = None, device: str = 'cuda'):
    """
    Load VGGT model from HuggingFace or local path using VGGT.from_pretrained().
    
    This function loads VGGT model using the official VGGT.from_pretrained() method
    which automatically downloads model weights from HuggingFace the first time.
    
    Parameters
    ----------
    model_name : str, optional
        HuggingFace model identifier (e.g., 'facebook/VGGT-1B') or local path.
        If None, defaults to 'facebook/VGGT-1B'.
    device : str
        Device to load model on ('cuda' or 'cpu')
    
    Returns
    -------
    VGGT
        Loaded VGGT model in eval mode
    """
    try:
        # Default model name
        if model_name is None:
            model_name = "facebook/VGGT-1B"
        
        logger.info(f"Loading VGGT model from: {model_name}")
        
        # Load model using VGGT.from_pretrained() which handles HuggingFace Hub
        # This will automatically download the model weights the first time
        model = VGGT.from_pretrained(model_name)
        
        # Move to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        logger.info(f"VGGT model loaded successfully on {device}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load VGGT model: {e}")
        logger.error(f"Please ensure the model name '{model_name}' is correct or provide a valid local model path.")
        raise

@torch.no_grad()
def run_vggt_inference(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """
    Run VGGT inference on multi-view images to extract 3D information.
    
    VGGT expects images in format [S, 3, H, W] where S is sequence length (number of views),
    and values should be in [0, 1] range. The model returns a dictionary with:
    - 'depth': Depth maps with shape [B, S, H, W, 1]
    - 'depth_conf': Confidence scores for depth predictions
    - 'world_points': 3D world coordinates with shape [B, S, H, W, 3]
    - 'world_points_conf': Confidence scores for world points
    - 'pose_enc': Camera pose encoding with shape [B, S, 9]
    - 'images': Original input images
    
    Parameters
    ----------
    model : torch.nn.Module
        VGGT model in eval mode
    images : torch.Tensor
        Multi-view images with shape (num_views, channels, height, width)
        Values should be in [0, 1] range
    device : str
        Device to run inference on
    
    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary containing 3D information from VGGT:
        - 'depth': Depth maps (num_views, height, width, 1)
        - 'depth_conf': Depth confidence scores (num_views, height, width)
        - 'world_points': 3D world coordinates (num_views, height, width, 3)
        - 'world_points_conf': World points confidence (num_views, height, width)
        - 'pose_enc': Camera pose encoding (num_views, 9)
    """
    model.eval()
    
    # Move images to device
    images = images.to(device)
    
    # VGGT expects input in format [S, 3, H, W] where S is sequence length (number of views)
    # The model's forward method will add batch dimension internally if needed
    # So we pass images as [S, 3, H, W] directly
    
    # Determine dtype for mixed precision (bfloat16 on Ampere+ GPUs, float16 otherwise)
    if device == 'cuda' and torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if compute_capability >= 8 else torch.float16
    else:
        dtype = torch.float32
    
    with torch.no_grad():
        try:
            # Use mixed precision for better performance on GPU
            if device == 'cuda' and torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = model(images)
            else:
                predictions = model(images)
            
        except Exception as e:
            logger.error(f"Error during VGGT inference: {e}")
            logger.error(f"Input images shape: {images.shape}")
            logger.error("Please check that the model input format matches expectations.")
            raise
    
    return predictions

def save_pseudo_3d_data(
    pseudo_3d_data: Dict[str, torch.Tensor],
    output_path: Path,
    frame_id: str,
    video_id: str,
    metadata: Optional[Dict] = None
):
    """
    Save pseudo 3D data to disk for SSL training.
    
    Saves data in numpy format (.npy) which is compatible with the existing
    SSL training pipeline (similar to create_ibl_encoding.py).
    
    Parameters
    ----------
    pseudo_3d_data : Dict[str, torch.Tensor]
        Dictionary containing 3D information from VGGT
    output_path : Path
        Path to save the .npy file
    frame_id : str
        Frame identifier
    video_id : str
        Video identifier
    metadata : Dict, optional
        Additional metadata to save with the data
    """
    # Convert tensors to numpy arrays
    data_dict = {}
    for key, value in pseudo_3d_data.items():
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.cpu().numpy()
        else:
            data_dict[key] = value
    
    # Add metadata
    data_dict['metadata'] = {
        'frame_id': frame_id,
        'video_id': video_id,
        **(metadata or {})
    }
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as .npy file
    np.save(output_path, data_dict)
    logger.debug(f"Saved pseudo 3D data to: {output_path}")

@torch.no_grad()
def main():
    """
    Main function to run VGGT inference on multi-view dataset.
    
    This function:
    1. Loads the multi-view dataset (e.g., fly-anipose)
    2. Loads VGGT model
    3. Processes each multi-view frame through VGGT
    4. Saves pseudo 3D information for SSL training
    """
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Generate pseudo 3D information from multi-view images using VGGT'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Absolute path to multi-view dataset directory (e.g., data/ssl/fly-anipose)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory to save pseudo 3D data'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help='VGGT model name from HuggingFace or local path (e.g., facebook/VGGT-1B). Default: facebook/VGGT-1B'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run inference on (cuda or cpu)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("VGGT Pseudo 3D Generation for SSL Training")
    logger.info("=" * 60)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {args.model_name or 'Auto-detect from HuggingFace'}")
    logger.info(f"Device: {args.device}")
    
    # Check if data directory exists
    if not data_dir.is_dir():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Load dataset
    logger.info("Loading multi-view dataset...")
    dataset = MVTDataset(
        data_dir=str(data_dir),
        imgaug_pipeline=None  # No augmentation for inference
    )
    logger.info(f"Dataset loaded: {len(dataset)} frames")
    logger.info(f"Available views: {dataset.available_views}")
    
    # Load VGGT model
    logger.info("Loading VGGT model...")
    vggt_model = load_vggt_model(model_name=args.model_name, device=args.device)
    
    # Process dataset
    logger.info("Starting VGGT inference...")
    total_frames = 0
    successful_frames = 0
    failed_frames = 0
    
    # Iterate directly over dataset (no DataLoader)
    for idx in tqdm(range(len(dataset)), desc="Processing frames"):
        # Get item from dataset
        # MVTDataset returns MultiViewDict with:
        # - input_image: (num_views, channels, height, width)
        # - video_id: str
        # - frame_id: str
        # - idx: int
        # - input_view_paths: list[str] - list of image paths for each view
        item = dataset[idx]
        
        # Extract metadata
        video_id = str(item['video_id'])
        frame_id = str(item['frame_id'])
        input_view_paths = item['input_view_paths']  # list of image paths
        
        # Process this frame
        _process_single_frame(
            vggt_model=vggt_model,
            image_paths=input_view_paths,
            video_id=video_id,
            frame_id=frame_id,
            output_dir=output_dir,
            device=args.device
        )
        successful_frames += 1
        total_frames += 1
    
    # Save summary
    summary = {
        'total_frames': total_frames,
        'successful_frames': successful_frames,
        'failed_frames': failed_frames,
        'data_dir': str(data_dir),
        'output_dir': str(output_dir),
        'model_name': args.model_name or 'Auto-detected',
        'device': args.device,
    }
    
    summary_path = output_dir / 'inference_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("Inference Complete")
    logger.info("=" * 60)
    logger.info(f"Total frames: {total_frames}")
    logger.info(f"Successful: {successful_frames}")
    logger.info(f"Failed: {failed_frames}")
    logger.info(f"Summary saved to: {summary_path}")


def _process_single_frame(
    vggt_model: torch.nn.Module,
    image_paths: list,
    video_id: str,
    frame_id: str,
    output_dir: Path,
    device: str
):
    """
    Process a single multi-view frame through VGGT.
    
    This function:
    1. Uses VGGT's load_and_preprocess_images to preprocess images from paths
    2. Runs VGGT inference
    3. Saves the results
    
    Parameters
    ----------
    vggt_model : torch.nn.Module
        VGGT model
    image_paths : list
        List of image file paths (strings) for each view
    video_id : str
        Video identifier
    frame_id : str
        Frame identifier (without extension)
    output_dir : Path
        Output directory
    device : str
        Device to run inference on
    """
    if not image_paths:
        raise ValueError(f"Empty image_paths list for video_id={video_id}, frame_id={frame_id}")
    
    # Use VGGT's preprocessing function
    # This handles proper preprocessing according to VGGT requirements
    preprocessed_images = load_and_preprocess_images(image_paths)
    
    # Move to device
    preprocessed_images = preprocessed_images.to(device)

    # Run VGGT inference
    predictions = run_vggt_inference(
        model=vggt_model,
        images=preprocessed_images,
        device=device
    )

    pseudo_3d_data = {
        'depth': predictions['depth'].squeeze(0),
        'world_points': predictions['world_points'].squeeze(0),
        'pose_enc': predictions['pose_enc'].squeeze(0),
    }

    # resize the depth and world points to 224x224
    # interpolate expects (N, C, H, W) format, so we need to permute from (S, H, W, C) to (S, C, H, W)
    # For depth: shape is (S, H, W, 1) -> permute to (S, 1, H, W)
    depth_permuted = pseudo_3d_data['depth'].permute(0, 3, 1, 2)  # (S, H, W, 1) -> (S, 1, H, W)
    pseudo_3d_data['depth'] = torch.nn.functional.interpolate(depth_permuted, size=(224, 224), mode='bilinear')
    
    # For world_points: shape is (S, H, W, 3) -> permute to (S, 3, H, W)
    world_points_permuted = pseudo_3d_data['world_points'].permute(0, 3, 1, 2)  # (S, H, W, 3) -> (S, 3, H, W)
    pseudo_3d_data['world_points'] = torch.nn.functional.interpolate(world_points_permuted, size=(224, 224), mode='bilinear')
    
    # Save pseudo 3D data
    frame_id = frame_id.split('.')[0][3:]
    frame_id = '3d_'+frame_id
    # Structure: output_dir/video_id/frame_id.npy
    output_path = output_dir / video_id / f"{frame_id}.npy"
    view_list = [image_path.split('/')[-2] for image_path in image_paths]
    
    save_pseudo_3d_data(
        pseudo_3d_data=pseudo_3d_data,
        output_path=output_path,
        frame_id=frame_id,
        video_id=video_id,
        metadata={
            'view_list': view_list,
        }
    )

if __name__ == '__main__':
    main()


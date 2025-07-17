import os
import torch
import numpy as np
import random
import argparse
from pathlib import Path
import re
import matplotlib.pyplot as plt
from models.mae import VisionTransformer, MVVisionTransformer

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225] 

NAME_MODEL = {
    'mae': VisionTransformer,
    'mvmae': MVVisionTransformer,
    'ijepa': 'facebook/vit-mae-base',
}

def denormalize_image(image):
    """
    Denormalize image from ImageNet normalization back to [0, 1] range.
    
    Args:
        image (np.ndarray or torch.Tensor): Normalized image with shape (C, H, W) or (B, C, H, W)
        
    Returns:
        np.ndarray: Denormalized image in [0, 1] range
    """
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    
    # Convert to numpy if needed
    image = np.array(image)
    
    # Handle batch dimension
    if image.ndim == 4:
        # (B, C, H, W) -> (B, C, H, W)
        mean = np.array(_IMAGENET_MEAN).reshape(1, 3, 1, 1)
        std = np.array(_IMAGENET_STD).reshape(1, 3, 1, 1)
    else:
        # (C, H, W) -> (C, H, W)
        mean = np.array(_IMAGENET_MEAN).reshape(3, 1, 1)
        std = np.array(_IMAGENET_STD).reshape(3, 1, 1)
    
    # Denormalize: (image * std) + mean
    denormalized = image * std + mean
    
    # Clip to [0, 1] range
    denormalized = np.clip(denormalized, 0, 1)
    
    return denormalized

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='data/ssl')
    parser.add_argument('--frame_rate', type=int, default=10)
    parser.add_argument('--frame_size', type=tuple, default=(224, 224))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='mirror-mouse-separate')
    parser.add_argument('--author', type=str, default='Yanchen Wang')
    parser.add_argument('--config', type=str, default='configs/mae.yaml', help='config file (yaml)')
    return parser.parse_args()

def set_seed(seed):
    # set seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('seed set to {}'.format(seed))


def get_video_paths_by_id(directory_path):
    """
    Read all MP4 video paths under a directory and group them by unique video ID.
    
    Args:
        directory_path (str): Path to the directory containing MP4 files
        
    Returns:
        dict: Dictionary with video IDs as keys and view paths as values in Pathlib.Path format
              Format: {video_id: {view: path, ...}}
    
    Example:
        Input files: 180605_000_bot.mp4, 180605_000_top.mp4
        Output: {'180605_000': {'bot': 'path/to/180605_000_bot.mp4', 'top': 'path/to/180605_000_top.mp4'}}
    """
    video_paths = {}
    directory = Path(directory_path)
    
    # Find all MP4 files in the directory
    mp4_files = list(directory.glob('*.mp4'))
    
    for mp4_file in mp4_files:
        filename = mp4_file.name
        
        # Extract video ID and view from filename
        # Pattern: video_id_view.mp4 (e.g., 180605_000_bot.mp4)
        match = re.match(r'^(.+)_([^_]+)\.mp4$', filename)
        
        if match:
            video_id = match.group(1)  # e.g., "180605_000"
            view = match.group(2)      # e.g., "bot"
            
            if video_id not in video_paths:
                video_paths[video_id] = {}
            
            video_paths[video_id][view] = mp4_file.resolve()
    
    return video_paths


def get_anchor_view_paths(video_dict, anchor_view):
    """
    Extract only the anchor view paths from the video dictionary.
    
    Args:
        video_dict (dict): Dictionary returned by get_video_paths_by_id
        anchor_view (str): Name of the anchor view (e.g., 'bot', 'top')
        
    Returns:
        list: List of paths for the anchor view only
        
    Example:
        video_dict = {'180605_000': {'bot': 'path1', 'top': 'path2'}}
        anchor_paths = get_anchor_view_paths(video_dict, 'bot')
        # Returns: ['path1']
    """
    anchor_paths = []
    
    for video_id, views in video_dict.items():
        if anchor_view in views:
            anchor_paths.append(views[anchor_view])
    
    return anchor_paths


def get_all_views_for_anchor(anchor_path, video_dict):
    """
    Get all view paths for the video ID that contains the given anchor path.
    
    Args:
        anchor_path (Path): Path to the anchor video file
        video_dict (dict): Dictionary returned by get_video_paths_by_id
        
    Returns:
        dict: Dictionary of all views for the same video ID as the anchor
              Format: {view: path, ...}
        
    Example:
        anchor_path = Path('/path/to/180605_000_bot.mp4')
        video_dict = {'180605_000': {'bot': path1, 'top': path2}}
        all_views = get_all_views_for_anchor(anchor_path, video_dict)
        # Returns: {'bot': path1, 'top': path2}
    """
    # Extract video ID from anchor path
    anchor_filename = anchor_path.name
    match = re.match(r'^(.+)_([^_]+)\.mp4$', anchor_filename)
    
    if not match:
        raise ValueError(f"Anchor path {anchor_path} does not match expected format")
    
    video_id = match.group(1)
    
    # Return all views for this video ID
    if video_id in video_dict:
        return video_dict[video_id]
    else:
        raise ValueError(f"Video ID {video_id} not found in video dictionary")


def get_video_id_from_path(video_path):
    """
    Extract the unique video ID from a video file path.
    
    Args:
        video_path (Path or str): Path to the video file
        
    Returns:
        str: The unique video ID extracted from the filename
        
    Example:
        video_path = Path('/path/to/180605_000_bot.mp4')
        video_id = get_video_id_from_path(video_path)
        # Returns: '180605_000'
    """
    if isinstance(video_path, str):
        video_path = Path(video_path)
    
    filename = video_path.name
    match = re.match(r'^(.+)_([^_]+)\.mp4$', filename)
    
    if not match:
        raise ValueError(f"Video path {video_path} does not match expected format")
    
    return match.group(1)


def plot_example_images(batch, results_dict, recon_num=8, save_path=None):
    """
    Plot example images with three columns: input images, output images, and reconstructed images.
    
    Args:
        batch (dict): Batch dictionary containing 'input_image', 'output_image', 'input_view', 'output_view'
        results_dict (dict): Results dictionary containing 'reconstructions'
        recon_num (int): Number of examples to plot (default: 8)
        save_path (str, optional): Path to save the plot (if None, displays the plot)
    """
    # Extract images and views
    input_images = batch['input_image'][:recon_num]
    output_images = batch['output_image'][:recon_num]
    recon_images = results_dict['reconstructions'][:recon_num]
    input_views = batch['input_view'][:recon_num]
    output_views = batch['output_view'][:recon_num]
    
    # Convert tensors to numpy arrays and move to CPU if needed
    if torch.is_tensor(input_images):
        input_images = input_images.detach().cpu().numpy()
    if torch.is_tensor(output_images):
        output_images = output_images.detach().cpu().numpy()
    if torch.is_tensor(recon_images):
        recon_images = recon_images.detach().cpu().numpy()
    # show the min and max of the images
    print(f'Input images min: {input_images.min()}, max: {input_images.max()}')
    print(f'Output images min: {output_images.min()}, max: {output_images.max()}')
    print(f'Recon images min: {recon_images.min()}, max: {recon_images.max()}')
    
    # Create subplot
    fig, axes = plt.subplots(recon_num, 3, figsize=(12, 4 * recon_num))
    
    # If only one example, make axes 2D
    if recon_num == 1:
        axes = axes.reshape(1, -1)
    
    # Plot images
    for i in range(recon_num):
        # Input images (first column)
        axes[i, 0].imshow(denormalize_image(input_images[i]).transpose(1, 2, 0))
        axes[i, 0].set_title(f'Input: {input_views[i]}')
        axes[i, 0].axis('off')
        
        # Output images (second column)
        axes[i, 1].imshow(denormalize_image(output_images[i]).transpose(1, 2, 0))
        axes[i, 1].set_title(f'Output: {output_views[i]}')
        axes[i, 1].axis('off')
        
        # Reconstructed images (third column)
        axes[i, 2].imshow(denormalize_image(recon_images[i]).transpose(1, 2, 0))
        
        axes[i, 2].set_title('Reconstructed')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        # find directory of save_path
        # and create it if it doesn't exist
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


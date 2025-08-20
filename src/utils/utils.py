import os
import torch
import numpy as np
import random
import argparse
from pathlib import Path
import re
import matplotlib.pyplot as plt
from models.mae import VisionTransformer, MVVisionTransformer
from data.datasets import MVDataset, BaseDataset

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225] 

NAME_MODEL = {
    'mae': VisionTransformer,
    'mvmae': MVVisionTransformer,
    'ijepa': 'facebook/vit-mae-base',
}
NAME_DATASET = {
    'mv': MVDataset,
    'base': BaseDataset,
}

def get_experiment_name(config):
    model_name = config['model']['name']
    type_name = config['data']['name']
    dataset_name = config['data']['data_dir'].split('/')[-1]
    return f'{model_name}_{type_name}_{dataset_name}_pretrain'

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
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    parser.add_argument('--resume_from_best', action='store_true', help='resume from best model instead of last model')
    parser.add_argument('--litpose_config', type=str, default='configs/litpose/config_mirror-mouse-separate.yaml', help='config file (yaml)')
    parser.add_argument('--litpose_frame', type=int, default=100, help='num of litpose frames to train on')
    parser.add_argument('--mode', type=str, default='ft', help='mode to train on')
    parser.add_argument('--model', type=str, default='mae', help='model to train on')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--data_dir', type=str, default='data', help='path to data directory')
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
        if 'Cam-' in filename:
            # Pattern: video_id_Cam-view_*.mp4 (e.g., 05272019_fly1_0_R1C24_Cam-A_rot-ccw-0.06_sec.mp4)
            # fly-pose dataset
            match = re.match(r'^(.+)_Cam-([^_]+)_.*\.mp4$', filename)
        elif 'iblrig' in filename:
            # Pattern: *_view.downsampled.*.video_id.mp4
            # iblri dataset
            match = re.match(r'^(.+)_([^_]+)\.downsampled.*\.([^.]+)\.mp4$', filename)
        else:
            # Extract video ID and view from filename
            # mirror-mouse-separate dataset
            # Pattern: video_id_view.mp4 (e.g., 180605_000_bot.mp4)
            match = re.match(r'^(.+)_([^_]+)\.mp4$', filename)
        
        if match:
            if 'iblrig' in filename:
                # For iblrig pattern: group(1) = prefix, group(2) = view, group(3) = video_id
                video_id = match.group(3)  # e.g., "video_id"
                view = match.group(2)      # e.g., "view"
            else:
                # For other patterns: group(1) = video_id, group(2) = view
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
    if 'Cam-' in anchor_path.name:
        # fly-pose dataset
        match = re.match(r'^(.+)_Cam-([^_]+)_.*\.mp4$', anchor_path.name)
    elif 'iblrig' in anchor_path.name:
        # iblrig dataset
        match = re.match(r'^(.+)_([^_]+)\.downsampled.*\.([^.]+)\.mp4$', anchor_path.name)
    else:
        # mirror-mouse-separate dataset
        match = re.match(r'^(.+)_([^_]+)\.mp4$', anchor_path.name)
    
    if not match:
        raise ValueError(f"Anchor path {anchor_path} does not match expected format")
    if 'iblrig' in anchor_path.name:
        video_id = match.group(3)
    else:
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
    if 'iblrig' in filename:
        # iblrig dataset
        match = re.match(r'^(.+)_([^_]+)\.downsampled.*\.([^.]+)\.mp4$', filename)
    else:
        match = re.match(r'^(.+)_([^_]+)\.mp4$', filename)
    
    if not match:
        raise ValueError(f"Video path {video_path} does not match expected format")
    if 'iblrig' in filename:
        return match.group(3)
    else:
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
    if 'input_image' in batch:
        input_images = batch['input_image'][:recon_num]
        output_images = batch['output_image'][:recon_num]
        input_views = batch['input_view'][:recon_num]
        output_views = batch['output_view'][:recon_num]
    else:
        input_images = batch['image'][:recon_num]
        output_images = batch['image'][:recon_num]
        input_views = [''] * recon_num
        output_views = [''] * recon_num
    recon_images = results_dict['reconstructions'][:recon_num]
    
    
    # Convert tensors to numpy arrays and move to CPU if needed
    if torch.is_tensor(input_images):
        input_images = input_images.detach().cpu().numpy()
    if torch.is_tensor(output_images):
        output_images = output_images.detach().cpu().numpy()
    if torch.is_tensor(recon_images):
        recon_images = recon_images.detach().cpu().numpy()
    
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


def create_log_dir(experiment_name: str, base_dir: str = "logs") -> str:
    """
    Create a log directory with timestamp and experiment name.
    
    Args:
        experiment_name (str): Name of the experiment
        base_dir (str): Base directory for logs, defaults to "logs"
        
    Returns:
        str: Path to the created log directory
        
    Example:
        log_dir = create_log_dir("mae_pretrain")
        # Returns: "logs/2024_01_15_14_30_25_mae_pretrain"
    """
    import datetime
    
    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    # Create log directory name
    log_dir_name = f"{timestamp}_{experiment_name}"
    log_dir_path = os.path.join(base_dir, log_dir_name)
    
    # Create directories
    os.makedirs(log_dir_path, exist_ok=True)
    os.makedirs(os.path.join(log_dir_path, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(log_dir_path, "plots"), exist_ok=True)
    
    return log_dir_path


def load_checkpoint_for_resume(checkpoint_path: str, accelerator, model, optimizer, scheduler, logger):
    """
    Load checkpoint for resuming training.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        accelerator: Accelerator instance
        model: Model instance
        optimizer: Optimizer instance
        scheduler: Scheduler instance
        logger: Logger instance
        
    Returns:
        dict: Dictionary containing loaded state information
    """
    import json
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = accelerator.load_state(checkpoint_path)
    
    # Extract training state information
    training_state = {
        'epoch': 0,  # Default values
        'best_loss': float('inf'),
        'best_epoch': 0,
        'global_step': 0
    }
    
    # Try to load training state from a separate file
    checkpoint_dir = os.path.dirname(checkpoint_path)
    training_state_path = os.path.join(checkpoint_dir, "training_state.json")
    
    if os.path.exists(training_state_path):
        try:
            with open(training_state_path, 'r') as f:
                training_state = json.load(f)
            logger.info(f"Loaded training state: {training_state}")
        except Exception as e:
            logger.warning(f"Failed to load training state: {e}")
    
    logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
    return training_state


def save_training_config(config: dict, training_info: dict, log_dir: str, logger=None):
    """
    Save training configuration and computed training information to a JSON file.
    
    Args:
        config (dict): Original configuration dictionary
        training_info (dict): Computed training information (epochs, lr, batch_size, etc.)
        log_dir (str): Log directory path
        logger: Logger instance for info messages
        
    Returns:
        str: Path to the saved configuration file
    """
    import json
    import datetime
    
    # Create a comprehensive configuration dictionary
    training_config = {
        "original_config": config,
        "computed_training_info": training_info,
        "timestamp": datetime.datetime.now().isoformat(),
        "log_directory": log_dir
    }
    
    # Save to JSON file
    config_path = os.path.join(log_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(training_config, f, indent=2)
    
    if logger:
        logger.info(f"Training configuration saved to: {config_path}")
    
    return config_path


def save_training_config_summary(training_info: dict, log_dir: str, logger=None):
    """
    Save a human-readable summary of training configuration to a text file.
    
    Args:
        training_info (dict): Computed training information
        log_dir (str): Log directory path
        logger: Logger instance for info messages
        
    Returns:
        str: Path to the saved summary file
    """
    import datetime
    
    # Create a readable summary
    summary_lines = [
        "=" * 60,
        "TRAINING CONFIGURATION SUMMARY",
        "=" * 60,
        f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Experiment: {training_info.get('experiment_name', 'N/A')}",
        "",
        "MODEL CONFIGURATION:",
        f"  Model: {training_info.get('model_name', 'N/A')}",
        f"  Available Views: {training_info.get('available_views', 'N/A')}",
        f"  Number of Views: {training_info.get('num_views', 'N/A')}",
        "",
        "TRAINING PARAMETERS:",
        f"  Epochs: {training_info.get('epochs', 'N/A')}",
        f"  Total Steps: {training_info.get('total_steps', 'N/A')}",
        f"  Steps per Epoch: {training_info.get('steps_per_epoch', 'N/A')}",
        f"  Learning Rate: {training_info.get('learning_rate', 'N/A'):.2e}",
        f"  Weight Decay: {training_info.get('weight_decay', 'N/A')}",
        f"  Warmup Percentage: {training_info.get('warmup_percentage', 'N/A')}",
        "",
        "BATCH SIZE CONFIGURATION:",
        f"  Global Batch Size: {training_info.get('global_batch_size', 'N/A')}",
        f"  Local Batch Size: {training_info.get('local_batch_size', 'N/A')}",
        f"  World Size (Processes): {training_info.get('world_size', 'N/A')}",
        "",
        "DATASET INFORMATION:",
        f"  Dataset Size: {training_info.get('dataset_size', 'N/A')} samples",
        "",
        "OPTIMIZATION:",
        f"  Optimizer: {training_info.get('optimizer_type', 'N/A')}",
        f"  Scheduler: {training_info.get('scheduler_type', 'N/A')}",
        f"  Seed: {training_info.get('seed', 'N/A')}",
        "",
        "=" * 60
    ]
    
    # Save to text file
    summary_path = os.path.join(log_dir, "training_config_summary.txt")
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    if logger:
        logger.info(f"Training configuration summary saved to: {summary_path}")
    
    return summary_path


def save_environment_info(args, log_dir: str, logger=None):
    """
    Save command line arguments and environment information for reproducibility.
    
    Args:
        args: Command line arguments from argparse
        log_dir (str): Log directory path
        logger: Logger instance for info messages
        
    Returns:
        str: Path to the saved environment info file
    """
    import json
    import datetime
    import sys
    import platform
    
    # Collect environment information
    env_info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "command_line_args": vars(args),
        "python_version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "environment_variables": {
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER", ""),
        }
    }
    
    # Save to JSON file
    env_path = os.path.join(log_dir, "environment_info.json")
    with open(env_path, 'w') as f:
        json.dump(env_info, f, indent=2)
    
    if logger:
        logger.info(f"Environment information saved to: {env_path}")
    
    return env_path


def save_all_training_info(config: dict, training_info: dict, args, log_dir: str, logger=None):
    """
    Save all training-related information in an organized manner.
    
    This function saves:
    1. Complete training configuration (JSON)
    2. Human-readable training summary (TXT)
    3. Environment information (JSON)
    
    Args:
        config (dict): Original configuration dictionary
        training_info (dict): Computed training information
        args: Command line arguments
        log_dir (str): Log directory path
        logger: Logger instance for info messages
        
    Returns:
        dict: Dictionary containing paths to all saved files
    """
    saved_files = {}
    
    # Save complete training configuration
    saved_files['config'] = save_training_config(config, training_info, log_dir, logger)
    
    # Save human-readable summary
    saved_files['summary'] = save_training_config_summary(training_info, log_dir, logger)
    
    # Save environment information
    saved_files['environment'] = save_environment_info(args, log_dir, logger)
    
    if logger:
        logger.info("=" * 50)
        logger.info("ALL TRAINING INFORMATION SAVED")
        logger.info("=" * 50)
        logger.info(f"Configuration: {saved_files['config']}")
        logger.info(f"Summary: {saved_files['summary']}")
        logger.info(f"Environment: {saved_files['environment']}")
        logger.info("=" * 50)
    
    return saved_files


def get_resume_checkpoint_path(resume_path: str, resume_from_best: bool = False) -> str:
    """
    Get the appropriate checkpoint path for resuming training.
    
    Args:
        resume_path (str): Base path to the log directory or specific checkpoint
        resume_from_best (bool): Whether to resume from best model instead of last model
        
    Returns:
        str: Path to the checkpoint file
    """
    if os.path.isfile(resume_path):
        # Direct path to checkpoint file
        return resume_path
    
    # Assume it's a log directory path
    if resume_from_best:
        checkpoint_path = os.path.join(resume_path, "checkpoints", "best_model.pth")
    else:
        checkpoint_path = os.path.join(resume_path, "checkpoints", "last_model.pth")
    
    return checkpoint_path

def get_video_frame_num(video_path):
    """
    Get the number of frames in a video file.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
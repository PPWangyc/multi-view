import argparse
import copy
import os
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import Accelerator
from facemap.neural_prediction.neural_model import KeypointsNetwork
from ray import train, tune
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import r2_score as r2_score_sklearn
from tqdm import tqdm

from data.datasets import BaseDataset, MVDataset, MVTDataset, VideoDataset
from models.ijepa import IJEPA, vit_base
from models.mae import MVVisionTransformer, VisionTransformer
from models.videomae import VideoMAE
from models.mvt import MultiViewTransformer
from models.rrr import train_model_main
from utils.log_utils import get_logger
from utils.metric_utils import bits_per_spike, compute_varexp

logger = get_logger()

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225] 

NAME_MODEL = {
    'videomae': VideoMAE,
    'mae': VisionTransformer,
    'svmae': VisionTransformer,
    'mvmae': MVVisionTransformer,
    'svijepa': IJEPA,
    'ijepa': IJEPA,
    'mvt': MultiViewTransformer,
}
NAME_DATASET = {
    'mv': MVDataset,
    'mvt': MVTDataset,
    'base': BaseDataset,
    'video': VideoDataset,
}

def _std(arr):
    mean = np.mean(arr, axis=0) # (T, N)
    std = np.std(arr, axis=0) # (T, N)
    std = np.clip(std, 1e-8, None) # (T, N) 
    arr = (arr - mean) / std
    return arr, mean, std

def get_experiment_name(config):
    model_name = config['model']['name']
    model_type = config['model']['type']
    type_name = config['data']['name']
    dataset_name = config['data']['data_dir'].split('/')[-1]
    return f'{model_name}_{model_type}_{type_name}_{dataset_name}_pretrain'

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
    parser.add_argument('--eid', type=str, default=None, help='experiment id to train on')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--avail_views', type=str, nargs='+', default=['bot', 'top'], help='available views for training')
    parser.add_argument('--model_type', type=str, default='sv', help='litpose model type to train on')
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
    logger.info('seed set to {}'.format(seed))


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
        if batch['input_image'].ndim == 5:
            input_images = batch['input_image'][0]
            output_images = batch['output_image'][0]
            input_views = [input[0] for input in batch['input_view']]
            output_views = [output[0] for output in batch['output_view']]
            recon_num = input_images.shape[0]
            results_dict['reconstructions'] = results_dict['reconstructions'][0]
        else:
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


def load_checkpoint_for_resume(checkpoint_path: str, accelerator, model, optimizer=None, scheduler=None, logger=None):
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
    import datetime
    import json

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
    import datetime
    import json
    import platform
    import sys

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

def create_encoding_log(metadata):
    """
    Create a log directory for encoding based on metadata.
    
    Args:
        metadata (dict): Metadata dictionary containing configuration details
        
    Returns:
        str: Path to the created log directory
    """
    
    # Create log directory
    log_dir = os.path.join('logs', 'encoding', metadata['eid'])
    resume_path = metadata.get('resume', None)
    if resume_path is not None:
        resume_path = resume_path.split('/')[-2]

    # sub-metadata for logging
    log_dir = os.path.join(
        log_dir,
        f'views-{"".join(metadata["avail_views"])}', 
        f'model-{metadata["model"]}',
        f"resume-{resume_path}")
    os.makedirs(log_dir, exist_ok=True)

    return log_dir

def train_rrr(
    config,
    encoding_dict,
    test=False,
    tune=False,
):
    lr=config["lr"]
    l2 = 100
    n_comp = 3
    smooth_w = 2 # smooth window 2 seconds
    ground_truth = {}
    eid = encoding_dict['eid']
    # if test, use test set
    if test:
        modes = ['train', 'test']
        ground_truth[eid] = copy.deepcopy(encoding_dict["test"]["spike"])
    else:
        modes = ['train', 'val']
        ground_truth[eid] = copy.deepcopy(encoding_dict["val"]["spike"])
    data_dict = {
        eid: {
            "X": [],
            "y": [],
            "setup": {}
        } 
    }
    # gaussian filter
    for mode in modes:
        data_dict[eid]["y"].append(gaussian_filter1d(encoding_dict[mode]["spike"], smooth_w, axis=1))
        data_dict[eid]["X"].append(np.concatenate([encoding_dict[mode][view] for view in encoding_dict['avail_views']], axis=2)) # (num_trial, num_time, num_view * feature_dim)

    # standardize
    _, mean_X, std_X = _std(data_dict[eid]['X'][0])
    _, mean_y, std_y = _std(data_dict[eid]['y'][0])
    
    for i in range(2):
        K = data_dict[eid]["X"][i].shape[0]
        T = data_dict[eid]["X"][i].shape[1]
        data_dict[eid]["X"][i] = (data_dict[eid]["X"][i] - mean_X) / std_X
        if len(data_dict[eid]["X"][i].shape) == 2:
            data_dict[eid]["X"][i] = np.expand_dims(data_dict[eid]["X"][i], axis=0)
        # add bias
        data_dict[eid]["X"][i] = np.concatenate([data_dict[eid]["X"][i], np.ones((K, T, 1))], axis=2)
        data_dict[eid]["y"][i] = (data_dict[eid]["y"][i] - mean_y) / std_y
        logger.info(f"X shape with bias: {data_dict[eid]['X'][i].shape}, y shape: {data_dict[eid]['y'][i].shape}")
    data_dict[eid]["setup"]["mean_X_Tv"] = mean_X
    data_dict[eid]["setup"]["std_X_Tv"] = std_X
    data_dict[eid]["setup"]["mean_y_TN"] = mean_y
    data_dict[eid]["setup"]["std_y_TN"] = std_y
    
    logger.info("Training RRR")
    test_bps = []
    _train_data = {eid: data_dict[eid]}
    model, mse_val = train_model_main(
        train_data=_train_data,
        l2=l2,
        n_comp=n_comp,
        model_fname='tmp',
        save=False,
        lr=lr,
    )
    logger.info(f"Model {eid} trained")
    with torch.no_grad():
        _, _, pred_orig = model.predict_y_fr(data_dict, eid, 1)
    pred = pred_orig.cpu().numpy()
    threshold = 1e-3
    trial_len = 2.
    pred = np.clip(pred, threshold, None)
    # Replace any NaN values in pred with the threshold
    if np.any(np.isnan(pred)) :
        logger.warning(f"Contain NaN value, replace to {threshold}")
        pred = np.nan_to_num(pred, nan=threshold)
    num_trial, num_time, num_neuron = pred.shape
    gt_held_out = ground_truth[eid]
    mean_fr = gt_held_out.sum(1).mean(0) / trial_len
    keep_idxs = np.arange(len(mean_fr)).flatten()

    bps_result_list = []
    for n_i in tqdm(keep_idxs, desc='co-bps'):
        bps = bits_per_spike(
            pred[:, :, [n_i]],
            gt_held_out[:, :, [n_i]],
            threshold=threshold,
        )
        if np.isinf(bps):
            bps = np.nan
        bps_result_list.append(bps)
    co_bps = np.nanmean(bps_result_list)
    # calculate variance explained
    with torch.no_grad():
        _, y_norm, y_pred_norm = model.predict_y(data_dict, eid, 1)
    y_pred_norm = y_pred_norm.cpu().numpy()
    y_norm = y_norm.cpu().numpy()
    y_norm = y_norm.reshape(-1, num_neuron)
    y_pred_norm = y_pred_norm.reshape(-1, num_neuron)
    ven = compute_varexp(y_norm, y_pred_norm)
    ve = np.nanmean(ven)
    # calculate variance unexplained, r2
    try:
        r2 = r2_score_sklearn(y_norm, y_pred_norm)
    except Exception as e:
        logger.error(str(e))
        r2 = -100000
    
    logger.info(f"Co-BPS: {co_bps}")
    logger.info(f"r2: {r2}")
    logger.info(f"Variance Explained: {ve}")
    test_bps.append(co_bps)
    y_norm = y_norm.reshape(num_trial, num_time, num_neuron)
    y_pred_norm = y_pred_norm.reshape(num_trial, num_time, num_neuron)
    result = {
        'gt': gt_held_out,
        'pred': pred,
        'norm_gt': y_norm,
        'norm_pred': y_pred_norm,
        'mean_X': data_dict[eid]["setup"]["mean_X_Tv"],
        'std_X': data_dict[eid]["setup"]["std_X_Tv"],
        'mean_y': data_dict[eid]["setup"]["mean_y_TN"],
        'std_y': data_dict[eid]["setup"]["std_y_TN"],
        'bps': co_bps,
        'r2': r2,
        'eid': eid,
        've': ve,
    }
    if tune:
        train.report({"bps": co_bps, "r2": r2, "ve": ve}) # only report the last result eid
    else:
        return result

def train_rrr_with_tune(
        encoding_dict,
        num_samples=10,
):
    search_space = {
        "lr": tune.loguniform(5e-2, 2),
    }
    analysis = tune.run(
        tune.with_parameters(
            train_rrr, 
            encoding_dict=encoding_dict,
            test=False,
            tune=True,
        ),
        resources_per_trial={"cpu": 2, "gpu": 1},
        config=search_space,
        num_samples=num_samples,
        log_to_file=False,
    )
    best_config = analysis.get_best_config(metric="bps", mode="max")
    logger.info(f"best config: {best_config}")
    return train_rrr(
        config=best_config,
        encoding_dict=encoding_dict,
        test=True,
        tune=False,
    )

class Embed_Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        assert len(self.X) == len(self.y), "X and y should have the same trial length"
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_tcn(
        config,
        encoding_dict,
        test=False,
        tune=False,
        verbose=True,
    ):
    lr = config["lr"]
    wd = config["wd"]
    smoothing_penalty=0.5
    epochs=100
    annealing_steps=2
    trial_len=2
    anneal_epochs = epochs - 50 * np.arange(1, annealing_steps + 1)
    threshold = 1e-3
    eid = encoding_dict['eid']
    avail_views = encoding_dict['avail_views']
    accelerator = Accelerator()
    result = {}

    test_mode = 'test' if test else 'val'

    train_X = np.concatenate([encoding_dict['train'][view] for view in avail_views], axis=2)
    train_y = encoding_dict['train']["spike"]
    test_X = np.concatenate([encoding_dict[test_mode][view] for view in avail_views], axis=2)
    test_y = encoding_dict[test_mode]["spike"]
    # copy gt test spike
    test_y_gt = copy.deepcopy(test_y)
    # gaussian filter
    train_y = gaussian_filter1d(train_y, trial_len, axis=1)
    test_y = gaussian_filter1d(test_y, trial_len, axis=1)
    # norm
    _, mean_X, std_X = _std(train_X)
    _, mean_y, std_y = _std(train_y)
    train_X = (train_X - mean_X) / std_X
    test_X = (test_X - mean_X) / std_X
    train_y = (train_y - mean_y) / std_y
    test_y = (test_y - mean_y) / std_y
    embed_size = train_X.shape[-1]
    num_neuron = train_y.shape[-1]
    train_dataset = Embed_Dataset(train_X, train_y)
    test_dataset = Embed_Dataset(test_X, test_y)
    n_test = len(test_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    model = KeypointsNetwork(
        n_in=embed_size,
        n_out=num_neuron,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=wd
    )
    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )
    for epoch in range(epochs):
        model.train()
        if epoch in anneal_epochs:
            logger.info("annealing learning rate") if verbose else None
            optimizer.param_groups[0]["lr"] /= 10.0
        for batch in train_loader:
            X, y = batch
            y_pred = model(
                x=X
            )[0]
            loss = ((y_pred - y) ** 2).mean()
            loss += (
                smoothing_penalty
                * (torch.diff(model.core.features[1].weight) ** 2).sum()
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0 and verbose:
            model.eval()
            test_y_pred = []
            test_y = []
            with torch.no_grad():
                for batch in test_loader:
                    X, y = batch
                    y_pred = model(
                        x=X
                    )[0]
                    test_y_pred.append(y_pred)
                    test_y.append(y)
            test_y_pred = torch.cat(test_y_pred, axis=0)
            test_y = torch.cat(test_y, axis=0)
            test_y_pred = test_y_pred.reshape(-1, num_neuron)
            test_y = test_y.reshape(-1, num_neuron)
            ve = compute_varexp(test_y, test_y_pred).mean()
            logger.info(f"Epoch: {epoch}, VE: {ve}")

    model.eval()
    test_y_pred = []
    test_y = []
    with torch.no_grad():
        for batch in test_loader:
            X, y = batch
            y_pred = model(
                x=X
            )[0]
            test_y_pred.append(y_pred)
            test_y.append(y)
    test_y_pred = torch.cat(test_y_pred, axis=0).cpu().numpy()
    test_y = torch.cat(test_y, axis=0).cpu().numpy()
    # reshape to (N * T, Neuorn)
    test_y_pred = test_y_pred.reshape(-1, num_neuron)
    test_y = test_y.reshape(-1, num_neuron)
    # calculate variance explained
    ve = compute_varexp(test_y, test_y_pred).mean()
    # calculate variance unexplained, r2
    r2 = r2_score_sklearn(test_y, test_y_pred)
    # reshape to (N, T, Neuron)
    test_y_pred = test_y_pred.reshape(n_test, -1, num_neuron)
    test_y = test_y.reshape(n_test, -1, num_neuron)
    norm_test_y, norm_test_y_pred = copy.deepcopy(test_y), copy.deepcopy(test_y_pred)
    # denormalize
    test_y_pred = test_y_pred * std_y + mean_y
    test_y_pred = np.clip(test_y_pred, threshold, None)
    # Replace any NaN values in pred with the threshold
    if np.any(np.isnan(test_y_pred)) :
        logger.warning(f"Contain NaN value, replace to {threshold}")
        test_y_pred = np.nan_to_num(test_y_pred, nan=threshold)
    # calculate co-bps
    bps_result_list = []
    for i in range(num_neuron):
        bps = bits_per_spike(
            test_y_pred[:,:,[i]],
            test_y_gt[:,:,[i]], # gt spike, without gaussian filter and normalization
            threshold=threshold
        )
        if np.isinf(bps):
            bps = np.nan
        bps_result_list.append(bps)
    co_bps = np.nanmean(bps_result_list)
    logger.info(f"Co-BPS: {co_bps}, R2: {r2}, VE: {ve}")
    result = {
        'gt': test_y_gt,
        'pred': test_y_pred,
        'norm_gt': norm_test_y,
        'norm_pred': norm_test_y_pred,
        'mean_X': mean_X,
        'std_X': std_X,
        'mean_y': mean_y,
        'std_y': std_y,
        'bps': co_bps,
        'r2': r2,
        'eid': eid,
        've': ve,
    }
    if tune:
        train.report({"bps": co_bps, "r2": r2, "ve": ve}) # only report the last result eid
    else:       
        return result

def train_tcn_with_tune(
        encoding_dict,
        num_samples=10,
    ):
    search_space = {
        "lr": tune.loguniform(1e-4, 3e-3),
        "wd": 1e-4,
    }
    analysis = tune.run(
        tune.with_parameters(
            train_tcn,
            encoding_dict=encoding_dict,
            test=False,
            tune=True,
            verbose=False,
        ),
        resources_per_trial={"cpu": 2, "gpu": 1},
        config=search_space,
        num_samples=num_samples,
    )
    best_config = analysis.get_best_config(metric="bps", mode="max")
    logger.info(f"Best config: {best_config}")
    # test data_dict, remove the 2nd last element of X and y since it is the validation set
    return train_tcn(
        config=best_config,
        encoding_dict=encoding_dict,
        test=True,
        tune=False,
        verbose=True,
    )

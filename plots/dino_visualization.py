"""
Visualize Pretrained DINOv3 models on animal videos
"""

import sys
from pathlib import Path
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import yaml
import safetensors

from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from torchvision.transforms import v2
from sklearn.decomposition import PCA
from scipy.ndimage import zoom

# Add src directory to Python path
# Find project root by looking for src directory
current = Path().resolve()
while current != current.parent:
    src_dir = current / 'src'
    if src_dir.exists() and (src_dir / 'models').exists():
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        break
    current = current.parent

from models.mvt import MultiViewTransformer
from data.datasets import MVTDataset
# Plot reconstructions and masked ground truth using the utility function
from utils.viz_utils import plot_mvt_reconstructions
from utils.utils import set_seed

set_seed(42)

BASE_DIR = Path('/data/Projects/multi-view')    
DATA_DIR = os.path.join(BASE_DIR, 'data', 'chickadee-crop', 'videos_new')
model_name = 'dino'
views = ['lBack', 'rBack', 'lFront', 'rFront']
video_id = 'PRL43_200701_142147'
OUTPUT_DIR = os.path.join(BASE_DIR, 'plots', f'{model_name}_visualization_{video_id}')
os.makedirs(OUTPUT_DIR, exist_ok=True)


NUM_FRAMES = 300
video_paths = []
for view in views:
    video_paths.append(os.path.join(DATA_DIR, video_id + f'_{view}.short.mp4'))

# Load frames from video
original_frames_dict = {}
for video_path in video_paths:
    video_reader = cv2.VideoCapture(video_path)
    total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    original_frames = []
    for i in range(NUM_FRAMES):
        ret, frame = video_reader.read()
        if not ret:
            raise ValueError(f"Failed to read frame {i} from {video_path}")
        original_frames.append(frame)
    video_reader.release()
    original_frames_dict[video_path] = original_frames
    print(f"Loaded {len(original_frames)} frames from {video_path}")

# Setup device and DINO model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
dino_model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
dino_model.to(device)
dino_model.eval()

def make_transform(resize_size: int = 256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

transform = make_transform(resize_size=256)

def process_frame_with_dino(frame, frame_idx, video_path, view_name):
    """Process a single frame with DINO, with caching."""
    # Create cache directory for this video/view
    view_output_dir = os.path.join(OUTPUT_DIR, view_name)
    os.makedirs(view_output_dir, exist_ok=True)
    
    cache_file = os.path.join(view_output_dir, f"frame_{frame_idx:05d}.npz")
    
    # Check if cached representation exists
    if os.path.exists(cache_file):
        print(f"Loading cached DINO representation for {view_name} frame {frame_idx}")
        data = np.load(cache_file)
        pc1 = data['pc1']
        pc2 = data['pc2']
        pc3 = data['pc3']
        pca_rgb_upsampled = data['pca_rgb_upsampled']
        resized_frame = data['resized_frame']
        pca = PCA(n_components=3)
        pca.components_ = data['pca_components']
        pca.explained_variance_ratio_ = data['pca_explained_variance_ratio']
        pca.mean_ = data['pca_mean']
        return resized_frame, pc1, pc2, pc3, pca_rgb_upsampled, pca
    
    # Process frame with DINO
    print(f"Processing {view_name} frame {frame_idx} with DINO...")
    
    # Convert BGR to RGB for processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Transform and resize
    input_image = transform(frame_rgb)
    input_image = input_image.unsqueeze(0).to(device)
    h, w = input_image.shape[2:]
    
    # Resize original frame to match model input size
    resized_frame = cv2.resize(frame_rgb, (w, h))
    
    # Run DINO inference
    with torch.inference_mode():
        outputs = dino_model(pixel_values=input_image)
    
    features = outputs.last_hidden_state
    
    # Skip cls, reg tokens (first 5 tokens for DINOv3)
    features = features[:, 5:, :]
    
    # Convert features to numpy and reshape for PCA
    features_np = features.cpu().numpy()[0]  # [num_patches, hidden_dim]
    
    # Apply PCA to reduce to 3 components
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features_np)  # [num_patches, 3]
    
    # Reshape to spatial dimensions
    num_patches = features_pca.shape[0]
    patch_size = int(np.sqrt(num_patches))
    assert patch_size * patch_size == num_patches, f"Expected square number of patches, got {num_patches}"
    
    # Reshape each component to spatial grid
    pc1 = features_pca[:, 0].reshape(patch_size, patch_size)
    pc2 = features_pca[:, 1].reshape(patch_size, patch_size)
    pc3 = features_pca[:, 2].reshape(patch_size, patch_size)
    
    # Normalize each component to [0, 1] range
    pc1_norm = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)
    pc2_norm = (pc2 - pc2.min()) / (pc2.max() - pc2.min() + 1e-8)
    pc3_norm = (pc3 - pc3.min()) / (pc3.max() - pc3.min() + 1e-8)
    
    # Stack as RGB channels
    pca_rgb = np.stack([pc1_norm, pc2_norm, pc3_norm], axis=-1)
    
    # Upsample to match image size
    upsample_factor = resized_frame.shape[0] / patch_size
    pca_rgb_upsampled = zoom(pca_rgb, (upsample_factor, upsample_factor, 1), order=1)
    
    # Ensure values are in [0, 1] range
    pca_rgb_upsampled = np.clip(pca_rgb_upsampled, 0, 1)
    
    # Save to cache
    np.savez(cache_file,
             pc1=pc1, pc2=pc2, pc3=pc3,
             pca_rgb_upsampled=pca_rgb_upsampled,
             resized_frame=resized_frame,
             pca_components=pca.components_,
             pca_explained_variance_ratio=pca.explained_variance_ratio_,
             pca_mean=pca.mean_)
    
    return resized_frame, pc1, pc2, pc3, pca_rgb_upsampled, pca

def create_visualization_frame(resized_frame, pc1, pc2, pc3, pca_rgb_upsampled, view_name, add_labels=False):
    """Create a single visualization frame with all views."""
    h, w = resized_frame.shape[:2]
    
    # Normalize PC components for visualization (use same colormap range)
    pc1_viz = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)
    pc2_viz = (pc2 - pc2.min()) / (pc2.max() - pc2.min() + 1e-8)
    pc3_viz = (pc3 - pc3.min()) / (pc3.max() - pc3.min() + 1e-8)
    
    # Apply colormap to each PC component
    # matplotlib colormap can be called directly on 2D arrays
    colormap = plt.cm.RdBu_r
    pc1_colored = colormap(pc1_viz)[:, :, :3]  # Remove alpha channel, shape: (H, W, 3)
    pc2_colored = colormap(pc2_viz)[:, :, :3]
    pc3_colored = colormap(pc3_viz)[:, :, :3]
    
    # Upsample PC components to match frame size
    upsample_factor = h / pc1.shape[0]
    pc1_upsampled = zoom(pc1_colored, (upsample_factor, upsample_factor, 1), order=1)
    pc2_upsampled = zoom(pc2_colored, (upsample_factor, upsample_factor, 1), order=1)
    pc3_upsampled = zoom(pc3_colored, (upsample_factor, upsample_factor, 1), order=1)
    
    # Ensure values are in [0, 1] range
    pc1_upsampled = np.clip(pc1_upsampled, 0, 1)
    pc2_upsampled = np.clip(pc2_upsampled, 0, 1)
    pc3_upsampled = np.clip(pc3_upsampled, 0, 1)
    
    # Convert to uint8
    resized_frame_uint8 = (resized_frame * 255).astype(np.uint8) if resized_frame.max() <= 1.0 else resized_frame.astype(np.uint8)
    pc1_uint8 = (pc1_upsampled * 255).astype(np.uint8)
    pc2_uint8 = (pc2_upsampled * 255).astype(np.uint8)
    pc3_uint8 = (pc3_upsampled * 255).astype(np.uint8)
    pca_rgb_uint8 = (pca_rgb_upsampled * 255).astype(np.uint8)
    
    # Add column labels to each image
    def add_label(img, label):
        """Add a text label at the top center of an image (RGB format)."""
        img_labeled = img.copy()
        # Convert RGB to BGR for OpenCV operations
        img_bgr = cv2.cvtColor(img_labeled, cv2.COLOR_RGB2BGR)
        
        # Calculate text position (top center)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        x = (img_bgr.shape[1] - text_width) // 2
        y = text_height + 10
        
        # Add background rectangle for better visibility (black in BGR)
        cv2.rectangle(img_bgr, 
                     (x - 5, y - text_height - 5), 
                     (x + text_width + 5, y + baseline + 5), 
                     (0, 0, 0), -1)
        
        # Add text (white in BGR)
        cv2.putText(img_bgr, label, (x, y), font, font_scale, (255, 255, 255), thickness)
        
        # Convert back to RGB
        img_labeled = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return img_labeled
    
    # Add labels to each column only if requested (for first row)
    if add_labels:
        resized_frame_labeled = add_label(resized_frame_uint8, "Original")
        pc1_labeled = add_label(pc1_uint8, "PC1")
        pc2_labeled = add_label(pc2_uint8, "PC2")
        pc3_labeled = add_label(pc3_uint8, "PC3")
        pca_rgb_labeled = add_label(pca_rgb_uint8, "RGB")
    else:
        resized_frame_labeled = resized_frame_uint8
        pc1_labeled = pc1_uint8
        pc2_labeled = pc2_uint8
        pc3_labeled = pc3_uint8
        pca_rgb_labeled = pca_rgb_uint8
    
    # Concatenate horizontally: Original, PC1, PC2, PC3, RGB
    combined = np.hstack([resized_frame_labeled, pc1_labeled, pc2_labeled, pc3_labeled, pca_rgb_labeled])
    
    return combined

# Process all frames for all views
print(f"\nProcessing {NUM_FRAMES} frames...")
all_visualizations = []

for frame_idx in range(NUM_FRAMES):
    frame_visualizations = []
    
    for view_idx, video_path in enumerate(video_paths):
        # Extract view name from path like: PRL43_200617_131904_lBack.short.mp4
        basename = os.path.basename(video_path)
        # Find the view name (should be one of the views)
        view_name = None
        for v in views:
            if v in basename:
                view_name = v
                break
        if view_name is None:
            # Fallback: extract from filename pattern
            parts = basename.split('_')
            view_name = parts[-1].split('.')[0] if len(parts) > 1 else 'unknown'
        
        frame = original_frames_dict[video_path][frame_idx]
        
        resized_frame, pc1, pc2, pc3, pca_rgb_upsampled, pca = process_frame_with_dino(
            frame, frame_idx, video_path, view_name
        )
        
        # Add labels only for the first view (first row)
        add_labels = (view_idx == 0)
        vis_frame = create_visualization_frame(resized_frame, pc1, pc2, pc3, pca_rgb_upsampled, view_name, add_labels=add_labels)
        frame_visualizations.append(vis_frame)
    
    # Stack all views vertically
    combined_frame = np.vstack(frame_visualizations)
    all_visualizations.append(combined_frame)
    
    if (frame_idx + 1) % 50 == 0:
        print(f"Processed {frame_idx + 1}/{NUM_FRAMES} frames")

# Create video
print("\nCreating video...")
output_video_path = os.path.join(OUTPUT_DIR, f"{video_id}_dino_visualization.mp4")
h, w = all_visualizations[0].shape[:2]
fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

for vis_frame in all_visualizations:
    # Convert RGB to BGR for OpenCV
    vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
    video_writer.write(vis_frame_bgr)

video_writer.release()
print(f"Video saved to: {output_video_path}")


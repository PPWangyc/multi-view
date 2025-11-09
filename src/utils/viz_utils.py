"""Visualization utilities for masks and bounding boxes."""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def generate_colors(n: int, palette: str = 'tab20', seed: Optional[int] = None) -> List[Tuple[int, int, int]]:
    """Generate distinct colors for visualization using seaborn color palettes.
    
    Args:
        n: Number of colors to generate
        palette: Seaborn palette name (e.g., 'tab20', 'Set3', 'husl', 'colorblind')
                 'tab20' gives 20 distinct colors, cycles if n > palette size
        seed: Random seed for reproducibility (optional, only affects some palettes)
    
    Returns:
        List of (R, G, B) tuples in range [0, 255]
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get seaborn color palette (returns RGB tuples in 0-1 range)
    try:
        palette_colors = sns.color_palette(palette, n_colors=n)
    except ValueError:
        # Fallback to default palette if specified palette doesn't exist
        logger.warning(f"Palette '{palette}' not found, using 'tab20' instead")
        palette_colors = sns.color_palette('tab20', n_colors=n)
    
    # Convert RGB floats (0-1 range) to RGB integers (0-255)
    colors = []
    for color in palette_colors:
        # color is a tuple of floats in [0, 1] range
        color_rgb = tuple(int(c * 255) for c in color)
        colors.append(color_rgb)
    
    return colors


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.5
) -> np.ndarray:
    """Overlay a colored mask on an image.
    
    Args:
        image: Input image (H, W, 3) as numpy array, RGB format
        mask: Binary mask (H, W) as numpy array, dtype uint8
        color: (R, G, B) tuple for mask color
        alpha: Transparency of the mask overlay (0.0 to 1.0)
    
    Returns:
        Image with mask overlaid, RGB format
    """
    # Ensure mask is binary
    mask_binary = (mask > 0).astype(np.float32)
    
    # Create colored mask
    colored_mask = np.zeros_like(image, dtype=np.float32)
    colored_mask[:, :, 0] = color[0]  # R
    colored_mask[:, :, 1] = color[1]  # G
    colored_mask[:, :, 2] = color[2]  # B
    
    # Apply mask with alpha blending
    mask_3d = np.stack([mask_binary] * 3, axis=2)
    result = image.astype(np.float32) * (1 - alpha * mask_3d) + colored_mask * (alpha * mask_3d)
    
    return result.astype(np.uint8)


def draw_bbox_on_image(
    image: np.ndarray,
    bbox: List[float],
    color: Tuple[int, int, int],
    thickness: int = 2,
    label: Optional[str] = None
) -> np.ndarray:
    """Draw a bounding box on an image.
    
    Args:
        image: Input image (H, W, 3) as numpy array, RGB format
        bbox: Bounding box [x, y, w, h] in COCO format
        color: (R, G, B) tuple for bbox color
        thickness: Line thickness
        label: Optional text label to display
    
    Returns:
        Image with bbox drawn, RGB format
    """
    x, y, w, h = bbox
    
    # Convert to integer coordinates
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    
    # Convert RGB to BGR for OpenCV
    color_bgr = (color[2], color[1], color[0])
    
    # Convert RGB image to BGR for OpenCV operations
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Draw rectangle
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color_bgr, thickness)
    
    # Draw label if provided
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        
        # Draw text background
        cv2.rectangle(
            image_bgr,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color_bgr,
            -1
        )
        
        # Draw text
        cv2.putText(
            image_bgr,
            label,
            (x1, y1 - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),  # White text
            text_thickness
        )
    
    # Convert back to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    return image_rgb


def overlay_mask_and_bbox(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: List[float],
    color: Tuple[int, int, int],
    mask_alpha: float = 0.5,
    bbox_thickness: int = 2,
    label: Optional[str] = None
) -> np.ndarray:
    """Overlay mask and bbox on an image.
    
    Args:
        image: Input image (H, W, 3) as numpy array, RGB format
        mask: Binary mask (H, W) as numpy array, dtype uint8
        bbox: Bounding box [x, y, w, h] in COCO format
        color: (R, G, B) tuple for mask and bbox color
        mask_alpha: Transparency of the mask overlay
        bbox_thickness: Line thickness for bbox
        label: Optional text label to display
    
    Returns:
        Image with mask and bbox overlaid, RGB format
    """
    # First overlay the mask
    result = overlay_mask_on_image(image, mask, color, alpha=mask_alpha)
    
    # Then draw the bbox
    result = draw_bbox_on_image(result, bbox, color, thickness=bbox_thickness, label=label)
    
    return result


def overlay_multiple_masks_and_bboxes(
    image: np.ndarray,
    masks: List[np.ndarray],
    bboxes: List[List[float]],
    colors: List[Tuple[int, int, int]],
    mask_alpha: float = 0.5,
    bbox_thickness: int = 2,
    labels: Optional[List[str]] = None
) -> np.ndarray:
    """Overlay multiple masks and bboxes on an image.
    
    Args:
        image: Input image (H, W, 3) as numpy array, RGB format
        masks: List of binary masks, each (H, W) numpy array, dtype uint8
        bboxes: List of bboxes, each [x, y, w, h] in COCO format
        colors: List of (R, G, B) tuples for each mask/bbox color
        mask_alpha: Transparency of the mask overlay
        bbox_thickness: Line thickness for bboxes
        labels: Optional list of text labels for each bbox
    
    Returns:
        Image with all masks and bboxes overlaid, RGB format
    """
    if len(masks) != len(bboxes) or len(masks) != len(colors):
        raise ValueError(
            f"Number of masks ({len(masks)}), bboxes ({len(bboxes)}), "
            f"and colors ({len(colors)}) must match"
        )
    
    result = image.copy()
    
    # Overlay each mask and bbox
    for i, (mask, bbox, color) in enumerate(zip(masks, bboxes, colors)):
        label = labels[i] if labels and i < len(labels) else None
        result = overlay_mask_and_bbox(
            result,
            mask,
            bbox,
            color,
            mask_alpha=mask_alpha,
            bbox_thickness=bbox_thickness,
            label=label
        )
    
    return result


def add_text_label(
    image: np.ndarray,
    text: str,
    position: str = 'top-left',
    font_scale: float = 1.0,
    thickness: int = 2,
    bg_color: Optional[Tuple[int, int, int]] = None,
    text_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """Add a text label to an image.
    
    Args:
        image: Input image (H, W, 3) as numpy array, RGB format
        text: Text to display
        position: Position of text ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        font_scale: Font scale
        thickness: Text thickness
        bg_color: Optional background color (R, G, B). If None, no background
        text_color: Text color (R, G, B), default white
    
    Returns:
        Image with text label, RGB format
    """
    h, w = image.shape[:2]
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Determine position
    if position == 'top-left':
        x = 10
        y = text_height + 10
    elif position == 'top-right':
        x = w - text_width - 10
        y = text_height + 10
    elif position == 'bottom-left':
        x = 10
        y = h - baseline - 10
    elif position == 'bottom-right':
        x = w - text_width - 10
        y = h - baseline - 10
    else:
        x = 10
        y = text_height + 10
    
    # Draw background if specified
    if bg_color:
        bg_color_bgr = (bg_color[2], bg_color[1], bg_color[0])  # RGB to BGR
        cv2.rectangle(
            image_bgr,
            (x - 5, y - text_height - 5),
            (x + text_width + 5, y + baseline + 5),
            bg_color_bgr,
            -1
        )
    
    # Draw text
    text_color_bgr = (text_color[2], text_color[1], text_color[0])  # RGB to BGR
    cv2.putText(
        image_bgr,
        text,
        (x, y),
        font,
        font_scale,
        text_color_bgr,
        thickness
    )
    
    # Convert back to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    return image_rgb


def stack_frames_vertically(
    frame_top: np.ndarray,
    frame_bottom: np.ndarray
) -> np.ndarray:
    """Stack two frames vertically (top frame on top, bottom frame on bottom).
    
    Args:
        frame_top: Top frame (H, W, 3) as numpy array, RGB format
        frame_bottom: Bottom frame (H, W, 3) as numpy array, RGB format
    
    Returns:
        Stacked frame (2*H, W, 3) if same width, otherwise resized to match width
    """
    h1, w1 = frame_top.shape[:2]
    h2, w2 = frame_bottom.shape[:2]
    
    # Resize to match width if needed
    if w1 != w2:
        # Resize both to the larger width
        target_w = max(w1, w2)
        frame_top = cv2.resize(frame_top, (target_w, h1))
        frame_bottom = cv2.resize(frame_bottom, (target_w, h2))
    
    # Stack vertically
    stacked = np.vstack([frame_top, frame_bottom])
    
    return stacked


def create_video_from_frames(
    frames: List[np.ndarray],
    output_path: Path,
    fps: int = 30
) -> None:
    """Create a video from a list of frames.
    
    Args:
        frames: List of frames, each as (H, W, 3) numpy array in RGB format
        output_path: Path to save the output video
        fps: Frames per second
    """
    if len(frames) == 0:
        logger.warning("No frames provided for video creation")
        return
    
    # Get frame dimensions
    h, w = frames[0].shape[:2]
    
    # Create video writer (OpenCV uses BGR)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    logger.info(f"Video saved to: {output_path}")


def create_visualization_video(
    frame_paths: List[str],
    masks: List[np.ndarray],
    bboxes: List[List[float]],
    output_path: Path,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    fps: int = 30,
    mask_alpha: float = 0.5,
    bbox_thickness: int = 2,
    labels: Optional[List[str]] = None
) -> None:
    """Create a video with masks and bboxes overlaid on frames.
    
    Args:
        frame_paths: List of paths to frame images
        masks: List of binary masks, each (H, W) numpy array
        bboxes: List of bboxes, each [x, y, w, h] in COCO format
        output_path: Path to save the output video
        colors: Optional list of colors for each mask/bbox (if None, auto-generate).
               If single color provided for all frames, it will be used for all.
        fps: Frames per second
        mask_alpha: Transparency of mask overlay
        bbox_thickness: Line thickness for bboxes
        labels: Optional list of labels for each frame (if single label, used for all)
    """
    if len(frame_paths) == 0:
        logger.warning("No frame paths provided")
        return
    
    if len(masks) != len(frame_paths) or len(bboxes) != len(frame_paths):
        raise ValueError(
            f"Number of frames ({len(frame_paths)}), masks ({len(masks)}), "
            f"and bboxes ({len(bboxes)}) must match"
        )
    
    # Handle colors - if single color or None, generate/extend
    if colors is None:
        colors = generate_colors(len(masks))
    elif len(colors) == 1:
        # Single color for all frames
        colors = colors * len(frame_paths)
    elif len(colors) != len(frame_paths):
        # Recycle colors if not enough
        colors = [colors[i % len(colors)] for i in range(len(frame_paths))]
    
    # Handle labels - if single label, use for all frames
    if labels is not None and not isinstance(labels, list):
        labels = [labels] * len(frame_paths)
    elif labels is not None and len(labels) == 1:
        labels = labels * len(frame_paths)
    
    # Process each frame
    visualized_frames = []
    for i, frame_path in enumerate(frame_paths):
        # Load image
        img = Image.open(frame_path).convert('RGB')
        img_array = np.array(img)
        
        # Get mask and bbox for this frame
        mask = masks[i]
        bbox = bboxes[i]
        color = colors[i]
        label = labels[i] if labels else None
        
        # Overlay mask and bbox
        visualized_frame = overlay_mask_and_bbox(
            img_array,
            mask,
            bbox,
            color,
            mask_alpha=mask_alpha,
            bbox_thickness=bbox_thickness,
            label=label
        )
        
        visualized_frames.append(visualized_frame)
    
    # Create video
    create_video_from_frames(visualized_frames, output_path, fps=fps)


def plot_mvt_reconstructions(
    reconstructions: Union[np.ndarray, torch.Tensor],
    mask: Union[np.ndarray, torch.Tensor],
    output_image: Union[np.ndarray, torch.Tensor],
    output_mod: List[str],
    avail_views: Optional[List[str]] = None,
    patch_size: int = 16,
    imagenet_mean: Optional[np.ndarray] = None,
    imagenet_std: Optional[np.ndarray] = None,
    mask_color: float = 0.5,
    figsize: Optional[Tuple[int, int]] = None,
    show_stats: bool = True,
    return_fig: bool = False
) -> Optional[plt.Figure]:
    """Plot MVT reconstructions with ground truth visualization based on output_mod.
    
    Creates visualizations based on output_mod:
    - If output_mod = ['rgb']: 3 rows (RGB GT, RGB Masked GT, RGB reconstruction)
    - If output_mod = ['rgb', 'depth']: 6 rows (RGB GT, RGB Masked GT, RGB recon, Depth GT, Depth Masked GT, Depth recon)
    - Ignores 'world_points' in output_mod
    
    Args:
        reconstructions: Reconstructed images, shape (num_views, channels, height, width)
                        Can be numpy array or torch tensor
        mask: Binary mask indicating which patches are masked, shape (num_views * num_patches,)
              where 1 = masked, 0 = visible. Can be numpy array or torch tensor
        output_image: Ground truth output images, shape (num_views, channels, height, width)
                     Can be numpy array or torch tensor
        output_mod: List of output modes, e.g., ['rgb'] or ['rgb', 'depth']
        avail_views: Optional list of view names for titles (e.g., ['view0', 'view1', ...])
        patch_size: Size of each patch (default: 16)
        imagenet_mean: ImageNet mean for denormalization (default: [0.485, 0.456, 0.406])
        imagenet_std: ImageNet std for denormalization (default: [0.229, 0.224, 0.225])
        mask_color: Gray value for masked patches (0.0 to 1.0, default: 0.5)
        figsize: Optional figure size tuple (width, height). If None, auto-calculated
        show_stats: Whether to print mask statistics (default: True)
        return_fig: Whether to return the figure object (default: False)
    
    Returns:
        matplotlib Figure object if return_fig=True, else None
    """
    # Filter out 'world_points' from output_mod
    output_mod_filtered = [mode for mode in output_mod if mode != 'world_points']
    
    # Default ImageNet normalization parameters
    if imagenet_mean is None:
        imagenet_mean = np.array([0.485, 0.456, 0.406])
    if imagenet_std is None:
        imagenet_std = np.array([0.229, 0.224, 0.225])
    
    # Convert torch tensors to numpy
    if isinstance(reconstructions, torch.Tensor):
        reconstructions = reconstructions.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if isinstance(output_image, torch.Tensor):
        output_image = output_image.detach().cpu().numpy()
    
    # Get dimensions
    num_views, total_channels, height, width = reconstructions.shape
    
    # Parse channel indices based on output_mod
    channel_indices = {}
    current_channel = 0
    for mode in output_mod:
        if mode == 'rgb':
            channel_indices['rgb'] = (current_channel, current_channel + 3)
            current_channel += 3
        elif mode == 'depth':
            channel_indices['depth'] = (current_channel, current_channel + 1)
            current_channel += 1
        elif mode == 'world_points':
            # Skip world_points - don't add to channel_indices
            current_channel += 3
    
    # Calculate number of patches per view
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    num_patches_per_view = num_patches_h * num_patches_w
    
    # Reshape mask from (num_views * num_patches,) to (num_views, num_patches_per_view)
    mask_reshaped = mask.reshape(num_views, num_patches_per_view)
    
    # Reshape mask to patch grid for each view: (num_views, num_patches_h, num_patches_w)
    mask_patch_grid = mask_reshaped.reshape(num_views, num_patches_h, num_patches_w)
    
    # Expand mask to pixel level by repeating each patch
    mask_pixel = np.repeat(np.repeat(mask_patch_grid, patch_size, axis=1), patch_size, axis=2)
    
    # Denormalize output_image (only RGB channels use ImageNet normalization)
    denormalized_output = np.zeros_like(output_image)
    for v in range(num_views):
        for c in range(total_channels):
            if 'rgb' in channel_indices and channel_indices['rgb'][0] <= c < channel_indices['rgb'][1]:
                # RGB channels: use ImageNet normalization
                rgb_idx = c - channel_indices['rgb'][0]
                denormalized_output[v, c] = output_image[v, c] * imagenet_std[rgb_idx] + imagenet_mean[rgb_idx]
            else:
                # Depth or other channels: just clip to [0, 1] (assuming already normalized)
                denormalized_output[v, c] = np.clip(output_image[v, c], 0, 1)
    denormalized_output = np.clip(denormalized_output, 0, 1)
    
    # Denormalize reconstructions
    denormalized_recon = np.zeros_like(reconstructions)
    for v in range(num_views):
        for c in range(total_channels):
            if 'rgb' in channel_indices and channel_indices['rgb'][0] <= c < channel_indices['rgb'][1]:
                # RGB channels: use ImageNet normalization
                rgb_idx = c - channel_indices['rgb'][0]
                denormalized_recon[v, c] = reconstructions[v, c] * imagenet_std[rgb_idx] + imagenet_mean[rgb_idx]
            else:
                # Depth or other channels: just clip to [0, 1]
                denormalized_recon[v, c] = np.clip(reconstructions[v, c], 0, 1)
    denormalized_recon = np.clip(denormalized_recon, 0, 1)
    
    # Set default view names if not provided
    if avail_views is None:
        avail_views = [f'View {i+1}' for i in range(num_views)]
    
    # Determine number of rows based on output_mod_filtered
    num_rows = 0
    if 'rgb' in output_mod_filtered:
        num_rows += 3  # RGB GT, RGB Masked GT, and RGB recon
    if 'depth' in output_mod_filtered:
        num_rows += 3  # Depth GT, Depth Masked GT, and Depth recon
    
    # Calculate figure size if not provided
    if figsize is None:
        figsize = (3 * num_views, 3 * num_rows)
    
    # Create plot
    fig, axes = plt.subplots(num_rows, num_views, figsize=figsize)
    if num_views == 1:
        axes = axes.reshape(num_rows, 1)
    elif num_rows == 1:
        axes = axes.reshape(1, num_views)
    
    row_idx = 0
    
    # Plot RGB if present
    if 'rgb' in output_mod_filtered:
        rgb_start, rgb_end = channel_indices['rgb']
        
        # Extract RGB channels
        output_rgb = denormalized_output[:, rgb_start:rgb_end, :, :]  # (num_views, 3, H, W)
        recon_rgb = denormalized_recon[:, rgb_start:rgb_end, :, :]  # (num_views, 3, H, W)
        
        # Create masked RGB ground truth
        masked_rgb_gt = output_rgb.copy()
        for v in range(num_views):
            for c in range(3):
                masked_rgb_gt[v, c] = np.where(mask_pixel[v] == 1, mask_color, output_rgb[v, c])
        
        # Rearrange for plotting: (num_views, C, H, W) -> (num_views, H, W, C)
        output_rgb_plot = output_rgb.transpose(0, 2, 3, 1)
        recon_rgb_plot = recon_rgb.transpose(0, 2, 3, 1)
        masked_rgb_gt_plot = masked_rgb_gt.transpose(0, 2, 3, 1)
        
        # Row 1: RGB Ground Truth
        for i in range(num_views):
            axes[row_idx, i].imshow(output_rgb_plot[i])
            axes[row_idx, i].set_title(f'RGB GT: {avail_views[i]}', fontsize=10)
            axes[row_idx, i].axis('off')
        row_idx += 1
        
        # Row 2: RGB Masked Ground Truth
        for i in range(num_views):
            axes[row_idx, i].imshow(masked_rgb_gt_plot[i])
            axes[row_idx, i].set_title(f'RGB Masked GT: {avail_views[i]}', fontsize=10)
            axes[row_idx, i].axis('off')
        row_idx += 1
        
        # Row 3: RGB Reconstruction
        for i in range(num_views):
            axes[row_idx, i].imshow(recon_rgb_plot[i])
            axes[row_idx, i].set_title(f'RGB Recon: {avail_views[i]}', fontsize=10)
            axes[row_idx, i].axis('off')
        row_idx += 1
    
    # Plot Depth if present
    if 'depth' in output_mod_filtered:
        depth_start, depth_end = channel_indices['depth']
        
        # Extract depth channel
        output_depth = denormalized_output[:, depth_start:depth_end, :, :]  # (num_views, 1, H, W)
        recon_depth = denormalized_recon[:, depth_start:depth_end, :, :]  # (num_views, 1, H, W)
        
        # Create masked depth ground truth
        masked_depth_gt = output_depth.copy()
        for v in range(num_views):
            masked_depth_gt[v, 0] = np.where(mask_pixel[v] == 1, mask_color, output_depth[v, 0])
        
        # Rearrange for plotting: (num_views, 1, H, W) -> (num_views, H, W, 1)
        output_depth_plot = output_depth.transpose(0, 2, 3, 1)
        recon_depth_plot = recon_depth.transpose(0, 2, 3, 1)
        masked_depth_gt_plot = masked_depth_gt.transpose(0, 2, 3, 1)
        
        # Row 4: Depth Ground Truth
        for i in range(num_views):
            axes[row_idx, i].imshow(output_depth_plot[i].squeeze(), cmap='inferno')
            axes[row_idx, i].set_title(f'Depth GT: {avail_views[i]}', fontsize=10)
            axes[row_idx, i].axis('off')
        row_idx += 1
        
        # Row 5: Depth Masked Ground Truth
        for i in range(num_views):
            axes[row_idx, i].imshow(masked_depth_gt_plot[i].squeeze(), cmap='inferno')
            axes[row_idx, i].set_title(f'Depth Masked GT: {avail_views[i]}', fontsize=10)
            axes[row_idx, i].axis('off')
        row_idx += 1
        
        # Row 6: Depth Reconstruction
        for i in range(num_views):
            axes[row_idx, i].imshow(recon_depth_plot[i].squeeze(), cmap='inferno')
            axes[row_idx, i].set_title(f'Depth Recon: {avail_views[i]}', fontsize=10)
            axes[row_idx, i].axis('off')
        row_idx += 1
    
    plt.tight_layout()
    plt.show()
    
    # Print mask statistics if requested
    if show_stats:
        mask_ratio_per_view = mask_reshaped.mean(axis=1)
        print(f"\nMask statistics:")
        print(f"  Total patches per view: {num_patches_per_view}")
        print(f"  Mask ratio per view: {mask_ratio_per_view}")
        print(f"  Average mask ratio: {mask.mean():.3f} ({(mask.mean() * 100):.1f}% of patches masked)")
    
    if return_fig:
        return fig
    return None


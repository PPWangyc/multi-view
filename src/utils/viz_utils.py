"""Visualization utilities for masks and bounding boxes."""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import seaborn as sns
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


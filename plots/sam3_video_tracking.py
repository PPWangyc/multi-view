"""
SAM3 Video Tracking with Text Prompts

This script uses SAM3's video tracking API to track multiple animals in videos
using text prompts. It uses the transformers library for SAM3 video tracking.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from matplotlib.colors import hsv_to_rgb
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor, Sam3Model, Sam3Processor
from transformers.video_utils import load_video
from accelerate import Accelerator
from PIL import Image
import gc
import json

def get_video_info(video_path: str) -> Dict[str, float]:
    """
    Extract video information (total frames, duration, FPS).
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with 'total_frames', 'fps', and 'duration' keys
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0.0
    
    cap.release()
    
    return {
        'total_frames': total_frames,
        'fps': fps,
        'duration': duration
    }


def load_video_frames(video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """
    Load video frames efficiently, optionally limiting to first max_frames.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load (None for all)
        
    Returns:
        List of frames as numpy arrays in RGB format
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_count += 1
        
        # Stop if we've loaded max_frames
        if max_frames is not None and frame_count >= max_frames:
            break
    
    cap.release()
    
    return frames


def visualize_detected_objects(
    frame: np.ndarray,
    input_boxes: List[List[List[float]]],
    obj_ids: List[int],
    output_path: Optional[Path] = None
) -> np.ndarray:
    """
    Visualize detected objects with bounding boxes and labels.
    
    Args:
        frame: Frame as numpy array (H, W, 3) in RGB format
        input_boxes: Bounding boxes in 3-level nested format [[[x1, y1, x2, y2], ...]]
        obj_ids: List of object IDs corresponding to each box
        output_path: Optional path to save the visualization image
        
    Returns:
        Annotated frame (H, W, 3) in BGR format
    """
    vis_frame = frame.copy()
    if isinstance(vis_frame, np.ndarray):
        # Convert RGB to BGR for OpenCV
        vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
    else:
        vis_frame_bgr = np.array(vis_frame)
        if len(vis_frame_bgr.shape) == 3 and vis_frame_bgr.shape[2] == 3:
            vis_frame_bgr = cv2.cvtColor(vis_frame_bgr, cv2.COLOR_RGB2BGR)
    
    # Extract boxes from nested format
    # input_boxes format: [[[x1, y1, x2, y2], [x1, y1, x2, y2], ...]]
    all_boxes = input_boxes[0]  # Get the list of all boxes for the image
    
    # Draw bounding boxes and labels
    for i, obj_id in enumerate(obj_ids):
        # Get box coordinates for this object (ordered to match obj_ids)
        if i < len(all_boxes):
            box_coords = all_boxes[i]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = box_coords
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Generate color for this object
            hue = (obj_id * 0.618) % 1.0
            color = hsv_to_rgb([hue, 0.8, 0.9])
            color_bgr = tuple(int(c * 255) for c in color[::-1])  # RGB to BGR
            
            # Draw bounding box
            cv2.rectangle(vis_frame_bgr, (x1, y1), (x2, y2), color_bgr, 3)
            
            # Draw label
            label = f"ID:{obj_id}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_y = max(y1, label_size[1] + 10)
            
            # Draw label background
            cv2.rectangle(
                vis_frame_bgr,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0] + 10, label_y + 5),
                color_bgr,
                -1
            )
            
            # Draw label text
            cv2.putText(
                vis_frame_bgr,
                label,
                (x1 + 5, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
    
    # Save visualization if output path is provided
    if output_path is not None:
        cv2.imwrite(str(output_path), vis_frame_bgr)
    
    return vis_frame_bgr


def visualize_tracks(
    frame: np.ndarray,
    tracked_objects: Dict[int, Dict],
    track_colors: Dict[int, np.ndarray]
) -> np.ndarray:
    """
    Visualize tracked objects on frame with masks, bboxes, track IDs, and scores.
    
    Args:
        frame: Original frame (H, W, 3) in BGR format
        tracked_objects: Dictionary mapping track_id to detection info
        track_colors: Dictionary mapping track_id to RGB color
        
    Returns:
        Annotated frame (H, W, 3) in BGR format
    """
    vis_frame = frame.copy()
    h, w = frame.shape[:2]
    
    for track_id, obj_info in tracked_objects.items():
        color = track_colors[track_id]
        color_bgr = tuple(int(c) for c in color[::-1])  # RGB to BGR
        
        # Draw mask overlay
        if obj_info.get('mask') is not None:
            mask = obj_info['mask']
            if isinstance(mask, torch.Tensor):
                # Convert bfloat16 to float32 before numpy conversion
                if mask.dtype == torch.bfloat16:
                    mask = mask.cpu().float().numpy()
                else:
                    mask = mask.cpu().numpy()
            if len(mask.shape) == 3:
                mask = mask[0]  # Remove batch dimension if present
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                mask = mask.astype(np.uint8)
            
            # Create colored overlay
            overlay = vis_frame.copy()
            overlay[mask > 0] = color_bgr
            vis_frame = cv2.addWeighted(vis_frame, 0.6, overlay, 0.4, 0)
        
        # Draw bounding box (always draw if available)
        if 'box' in obj_info and obj_info['box'] is not None:
            box = obj_info['box']
            if isinstance(box, torch.Tensor):
                if box.dtype == torch.bfloat16:
                    box = box.cpu().float().numpy()
                else:
                    box = box.cpu().numpy()
            box = box.astype(int)
            x1, y1, x2, y2 = box
            # Draw bounding box with thicker line for better visibility
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color_bgr, 2)
        
        # Draw track ID and score
        score = obj_info.get('score', 0.0)
        if isinstance(score, torch.Tensor):
            score = score.cpu().item()
        label = f"ID:{track_id} ({score:.2f})"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Get position for label
        if 'box' in obj_info and obj_info['box'] is not None:
            x1, y1 = box[:2]
            label_y = max(y1, label_size[1] + 10)
        else:
            # If no box, use top-left corner
            x1, label_y = 10, label_size[1] + 10
        
        # Draw label background
        cv2.rectangle(
            vis_frame,
            (x1, label_y - label_size[1] - 5),
            (x1 + label_size[0] + 5, label_y + 5),
            color_bgr,
            -1
        )
        
        # Draw label text
        cv2.putText(
            vis_frame,
            label,
            (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
    
    return vis_frame


def extract_bbox_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract bounding box from mask in xyxy format.
    
    Args:
        mask: Binary mask (H, W) - can be float or uint8
        
    Returns:
        Bounding box in xyxy format [x1, y1, x2, y2] or None
    """
    if mask is None or mask.size == 0:
        return None
    
    # Convert to binary mask if needed
    if mask.dtype != np.uint8:
        # Normalize if needed
        if mask.max() > 1.0:
            mask = mask / 255.0
        # Threshold to create binary mask
        mask_binary = (mask > 0.5).astype(np.uint8)
    else:
        # If uint8, threshold at 127
        mask_binary = (mask > 127).astype(np.uint8) if mask.max() > 1 else mask
    
    # Check if mask has any foreground pixels
    if mask_binary.sum() == 0:
        return None
    
    # Find rows and columns with foreground pixels
    rows = np.any(mask_binary, axis=1)
    cols = np.any(mask_binary, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
    
    # Get bounding box coordinates
    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None
    
    y1, y2 = y_indices[0], y_indices[-1]
    x1, x2 = x_indices[0], x_indices[-1]
    
    # Ensure valid box (x1 < x2 and y1 < y2)
    if x1 >= x2 or y1 >= y2:
        return None
    
    return np.array([x1, y1, x2, y2], dtype=np.int32)


def detect_objects_with_text_prompt(
    frame: np.ndarray,
    text_prompt: str,
    device: torch.device,
    num_object: Optional[int] = None,
    threshold: float = 0.5
) -> Tuple[List[List[List[List[int]]]], List[List[List[int]]], List[int], List[List[List[float]]]]:
    """
    Use SAM3 text prompt to detect objects on a frame and convert to point inputs.
    
    Args:
        frame: Frame as numpy array (H, W, 3) in RGB format
        text_prompt: Text prompt for detection (e.g., "fishes")
        device: Device to run inference on
        num_object: Maximum number of objects to select (None for all)
        threshold: Confidence threshold for detections
        
    Returns:
        Tuple of (input_points, input_labels, obj_ids, input_boxes)
        - input_points: List of points for each object [[[[x, y]]], ...]
        - input_labels: List of labels [[[1]], ...] (all 1 for foreground)
        - obj_ids: List of object IDs [1, 2, ...]
        - input_boxes: List of bounding boxes in 3-level nested format [[[x1, y1, x2, y2]], ...]
    """
    # Convert frame to PIL Image
    if isinstance(frame, np.ndarray):
        frame_pil = Image.fromarray(frame)
    else:
        frame_pil = frame
    
    # Load SAM3 model and processor for text-based detection
    print(f"Loading SAM3 model for text-based detection...")
    sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
    sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")
    print("SAM3 model loaded for text detection\n")
    
    # Process inputs with text prompt
    print(f"Detecting objects with text prompt: '{text_prompt}'")
    inputs = sam3_processor(images=frame_pil, text=text_prompt, return_tensors="pt")
    
    # Convert inputs to device and match model dtype (bfloat16)
    # Only convert floating point tensors to bfloat16, keep integer tensors as is
    processed_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if v.dtype.is_floating_point:
                processed_inputs[k] = v.to(device, dtype=torch.bfloat16)
            else:
                processed_inputs[k] = v.to(device)
        else:
            processed_inputs[k] = v
    inputs = processed_inputs
    
    # Run inference
    with torch.no_grad():
        outputs = sam3_model(**inputs)
    
    # Post-process to get detections
    results = sam3_processor.post_process_instance_segmentation(
        outputs=outputs,
        threshold=threshold,
        mask_threshold=0.5,
        target_sizes=[frame_pil.size[::-1]]  # (height, width)
    )[0]
    
    # Extract boxes and scores
    # Convert to float32 before numpy conversion (numpy doesn't support bfloat16)
    boxes = results["boxes"].cpu().float().numpy()  # (N, 4) in xyxy format
    scores = results["scores"].cpu().float().numpy()  # (N,)
    
    print(f"Found {len(boxes)} objects with text prompt")
    
    # Select top num_object objects if specified
    if num_object is not None and len(boxes) > num_object:
        top_indices = np.argsort(scores)[::-1][:num_object]
        boxes = boxes[top_indices]
        scores = scores[top_indices]
        print(f"Selected top {num_object} objects by score")
    
    # Convert boxes to centroids (points) and format boxes for video tracker
    input_points = []
    input_labels = []
    obj_ids = []
    # Format: list[list[list[float]]] - 3 levels total
    # Level 1: Image level (one image) - outer list
    # Level 2: Box level (all boxes for that image) - middle list, each box is a list
    # Level 3: Box coordinates [x1, y1, x2, y2] - inner list
    box_coords_list = []
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        # Calculate centroid
        centroid_x = int((x1 + x2) / 2)
        centroid_y = int((y1 + y2) / 2)
        
        input_points.append([[[centroid_x, centroid_y]]])
        input_labels.append([[1]])  # All foreground points
        obj_ids.append(i + 1)  # Start from 1
        # Collect box coordinates as list of floats
        box_coords_list.append([float(x1), float(y1), float(x2), float(y2)])
    
    # Format as 3-level nested list: [[[x1, y1, x2, y2], [x1, y1, x2, y2], ...]]
    # Structure: [image_level[box_level[coordinates]]]
    # All boxes for one image, ordered to match obj_ids (first box -> first obj_id, etc.)
    input_boxes = [box_coords_list]
    
    print(f"Converted {len(input_points)} objects to point inputs\n")
    print(f"Number of obj_ids: {len(obj_ids)}, Number of boxes in input_boxes[0]: {len(input_boxes[0])}\n")
    print(f"Input boxes format (3 levels): {input_boxes}\n")
    
    # Clean up
    del sam3_model, sam3_processor, inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return input_points, input_labels, obj_ids, input_boxes


def process_sam3_video_outputs(
    outputs: Dict,
    frame_idx: int,
    selected_object_ids: Optional[List[int]] = None
) -> Dict[int, Dict]:
    """
    Process SAM3 video outputs and extract tracked objects.
    
    Args:
        outputs: SAM3 video predictor outputs (new format with masks dict)
        frame_idx: Current frame index
        selected_object_ids: List of object IDs to track (None for all)
        
    Returns:
        Dictionary mapping track_id (object_id) to object info (mask, box, score)
    """
    tracked_objects = {}
    
    if not outputs:
        return tracked_objects
    
    # New format: masks are organized by object_id in a dictionary
    if 'masks' in outputs and isinstance(outputs['masks'], dict):
        masks_dict = outputs['masks']
        scores_dict = outputs.get('scores', {})
        object_ids = outputs.get('object_ids', list(masks_dict.keys()))
        
        # Filter to selected object IDs if specified
        if selected_object_ids is not None:
            object_ids = [obj_id for obj_id in object_ids if obj_id in selected_object_ids]
        
        # Process each object
        for object_id in object_ids:
            if object_id not in masks_dict:
                continue
            
            mask = masks_dict[object_id]
            
            # Convert to numpy if tensor (convert bfloat16 to float32 first)
            if isinstance(mask, torch.Tensor):
                if mask.dtype == torch.bfloat16:
                    mask = mask.cpu().float().numpy()
                else:
                    mask = mask.cpu().numpy()
            
            # Extract bounding box from mask (always extract for visualization)
            # Ensure mask is in correct format for bbox extraction
            mask_for_bbox = mask.copy()
            
            # Handle different mask formats
            if len(mask_for_bbox.shape) > 2:
                # Remove extra dimensions
                mask_for_bbox = mask_for_bbox.squeeze()
                if len(mask_for_bbox.shape) > 2:
                    mask_for_bbox = mask_for_bbox[0]  # Take first channel if still 3D
            
            # Normalize mask values to [0, 1] range if needed
            if mask_for_bbox.max() > 1.0:
                mask_for_bbox = mask_for_bbox / 255.0
            
            # Extract bounding box
            box = extract_bbox_from_mask(mask_for_bbox)
            
            # Get score from scores_dict if available
            score = 1.0  # Default score
            if object_id in scores_dict:
                score_tensor = scores_dict[object_id]
                if isinstance(score_tensor, torch.Tensor):
                    if score_tensor.dtype == torch.bfloat16:
                        score = score_tensor.cpu().float().item()
                    else:
                        score = score_tensor.cpu().item()
                else:
                    score = float(score_tensor)
            
            tracked_objects[int(object_id)] = {
                'mask': mask,
                'box': box,
                'score': score
            }
    
    # Fallback: old format (for compatibility)
    else:
        object_ids = outputs.get('object_ids', None)
        scores = outputs.get('scores', None)
        boxes = outputs.get('boxes', None)
        masks = outputs.get('masks', None)
        
        if object_ids is None:
            return tracked_objects
        
        # Convert to numpy if tensors (handle bfloat16)
        if isinstance(object_ids, torch.Tensor):
            object_ids = object_ids.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            if scores.dtype == torch.bfloat16:
                scores = scores.cpu().float().numpy()
            else:
                scores = scores.cpu().numpy()
        if isinstance(boxes, torch.Tensor):
            if boxes.dtype == torch.bfloat16:
                boxes = boxes.cpu().float().numpy()
            else:
                boxes = boxes.cpu().numpy()
        if isinstance(masks, torch.Tensor):
            if masks.dtype == torch.bfloat16:
                masks = masks.cpu().float().numpy()
            else:
                masks = masks.cpu().numpy()
        
        # Filter to selected object IDs if specified
        if selected_object_ids is not None:
            selected_object_ids = np.array(selected_object_ids)
            mask_indices = np.isin(object_ids, selected_object_ids)
            object_ids = object_ids[mask_indices]
            if scores is not None:
                scores = scores[mask_indices]
            if boxes is not None:
                boxes = boxes[mask_indices]
            if masks is not None:
                masks = masks[mask_indices]
        
        # Process each object
        num_objects = len(object_ids)
        for obj_idx in range(num_objects):
            object_id = int(object_ids[obj_idx])
            score = float(scores[obj_idx]) if scores is not None and obj_idx < len(scores) else 1.0
            box = boxes[obj_idx] if boxes is not None and obj_idx < len(boxes) else None
            mask = masks[obj_idx] if masks is not None and obj_idx < len(masks) else None
            
            # Convert mask to numpy if tensor (handle bfloat16)
            if mask is not None and isinstance(mask, torch.Tensor):
                if mask.dtype == torch.bfloat16:
                    mask = mask.cpu().float().numpy()
                else:
                    mask = mask.cpu().numpy()
            
            # Extract box from mask if box is None
            if box is None and mask is not None:
                box = extract_bbox_from_mask(mask)
            
            tracked_objects[object_id] = {
                'mask': mask,
                'box': box,
                'score': score
            }
    
    return tracked_objects

def load_video_frames_streaming(video_path: str, max_frames: Optional[int] = None) -> np.ndarray:
    """
    Load video frames efficiently using streaming approach to minimize memory usage.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load (None for all)
        
    Returns:
        Numpy array of frames in RGB format
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    
    # Read first frame to get dimensions
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Could not read first frame from video")
    
    h, w = first_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    
    # Pre-allocate array for efficiency
    frames = np.zeros((total_frames, h, w, 3), dtype=np.uint8)
    frame_count = 0
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames[frame_count] = frame_rgb
        frame_count += 1
    
    cap.release()
    
    # Trim array if we read fewer frames than expected
    if frame_count < total_frames:
        frames = frames[:frame_count]
    
    return frames


def merge_videos(video_paths: List[str], output_path: str, fps: int) -> None:
    """
    Merge multiple video files into one.
    
    Args:
        video_paths: List of paths to video files to merge
        output_path: Path to save merged video
        fps: Frames per second for output video
    """
    import subprocess
    
    if len(video_paths) == 0:
        return
    
    if len(video_paths) == 1:
        # Just copy the single video
        import shutil
        shutil.copy(video_paths[0], output_path)
        return
    
    # Create file list for ffmpeg
    file_list_path = Path(output_path).parent / "video_list.txt"
    with open(file_list_path, 'w') as f:
        for video_path in video_paths:
            f.write(f"file '{Path(video_path).absolute()}'\n")
    
    # Use ffmpeg to concatenate videos
    cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', str(file_list_path),
        '-c', 'copy',
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Merged {len(video_paths)} videos into: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error merging videos: {e}")
        print(f"ffmpeg stderr: {e.stderr.decode() if e.stderr else 'N/A'}")
        raise
    finally:
        # Clean up file list
        if file_list_path.exists():
            file_list_path.unlink()


def process_video_clip(
    clip_frames: np.ndarray,
    clip_idx: int,
    global_frame_offset: int,
    model: Sam3TrackerVideoModel,
    processor: Sam3TrackerVideoProcessor,
    device: torch.device,
    output_path: Path,
    video_path: str,
    text_prompt: Optional[str] = None,
    num_object: Optional[int] = None,
    prev_clip_last_frame_tracking: Optional[Dict] = None,
    track_colors: Optional[Dict[int, np.ndarray]] = None,
    downsample_ratio: int = 1
) -> Tuple[Dict[int, np.ndarray], Dict, int]:
    """
    Process a single video clip (512 frames).
    
    Args:
        clip_frames: Frames for this clip (N, H, W, 3) in RGB
        clip_idx: Index of this clip (0-based)
        global_frame_offset: Global frame offset for this clip
        model: SAM3 tracker model
        processor: SAM3 tracker processor
        device: Device to run inference on
        output_path: Output directory path
        video_path: Path to original video file
        text_prompt: Text prompt for first clip detection
        num_object: Max number of objects to track
        prev_clip_last_frame_tracking: Tracking results from previous clip's last frame
        track_colors: Dictionary mapping track_id to color (will be updated)
        frame_skip: Frame skip factor
    Returns:
        Tuple of (updated_track_colors, last_frame_tracking, processed_frame_count)
        - updated_track_colors: Updated track colors dictionary
        - last_frame_tracking: Tracking results from last frame of this clip
        - processed_frame_count: Number of frames processed (excluding duplicate frame)
    """
    print(f"\n{'='*60}")
    print(f"Processing Clip {clip_idx + 1} ({len(clip_frames)} frames)")
    print(f"{'='*60}\n")
    
    # Initialize video session for this clip
    print(f"Initializing video session for clip {clip_idx + 1}...")
    inference_session = processor.init_video_session(
        video=clip_frames,
        inference_device=device,
        dtype=torch.bfloat16,
    )
    print("Video session initialized!\n")
    
    # Setup output directories
    outputs_dir = output_path / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    masks_dir = output_path / "masks"
    masks_dir.mkdir(exist_ok=True)
    clips_dir = output_path / "clips"
    clips_dir.mkdir(exist_ok=True)
    
    # Initialize tracking for this clip
    ann_frame_idx = 0
    
    if clip_idx == 0 or (clip_idx > 0 and prev_clip_last_frame_tracking is None):
        # First clip: detect objects using text prompt
        first_frame = clip_frames[0]
        input_points, input_labels, obj_ids, input_boxes = detect_objects_with_text_prompt(
            frame=first_frame,
            text_prompt=text_prompt,
            device=device,
            num_object=num_object,
            threshold=0.5
        )
        
        if len(input_points) == 0:
            raise ValueError(f"No objects detected with text prompt: '{text_prompt}'")
        
        print(f"Detected {len(input_points)} objects using text prompt")
        
        # Visualize detected objects on first frame (only for first clip)
        if clip_idx == 0:
            detection_vis_path = output_path / "detection_visualization.jpg"
            visualize_detected_objects(
                frame=first_frame,
                input_boxes=input_boxes,
                obj_ids=obj_ids,
                output_path=detection_vis_path
            )
            print(f"Saved detection visualization\n")
    else:
        # Subsequent clips: use previous clip's last frame tracking results
        if prev_clip_last_frame_tracking is None:
            raise ValueError("Previous clip tracking results required for clip > 0")
        
        # Extract boxes and object IDs from previous tracking
        obj_ids = list(prev_clip_last_frame_tracking['track_ids'])
        input_boxes = [[prev_clip_last_frame_tracking['bounding_boxes'][tid] 
                        for tid in obj_ids if prev_clip_last_frame_tracking['bounding_boxes'][tid] is not None]]
        
        # Filter out None boxes
        valid_indices = [i for i, box in enumerate(input_boxes[0]) if box is not None]
        obj_ids = [obj_ids[i] for i in valid_indices]
        input_boxes = [[input_boxes[0][i] for i in valid_indices]]
        
        if len(obj_ids) == 0:
            raise ValueError("No valid tracking results from previous clip")
        
        print(f"Initializing clip {clip_idx + 1} with {len(obj_ids)} objects from previous clip")
    
    # Add inputs to inference session
    processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=ann_frame_idx,
        obj_ids=obj_ids,
        input_boxes=input_boxes,
    )
    print(f"Added {len(obj_ids)} objects to tracking session\n")
    
    # Initialize track colors if not provided
    if track_colors is None:
        track_colors = {}
    for track_id in inference_session.obj_ids:
        if track_id not in track_colors:
            hue = (track_id * 0.618) % 1.0
            color = hsv_to_rgb([hue, 0.8, 0.9])
            track_colors[track_id] = np.array([int(c * 255) for c in color])
    
    # Segment objects on first frame
    print("Segmenting objects on first frame...")
    outputs = model(
        inference_session=inference_session,
        frame_idx=ann_frame_idx,
    )
    video_res_masks = processor.post_process_masks(
        [outputs.pred_masks],
        original_sizes=[[inference_session.video_height, inference_session.video_width]],
    )[0]
    print(f"Segmentation complete\n")
    
    # Setup video writer for this clip
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    clip_video_path = clips_dir / f"clip_{clip_idx:04d}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(clip_video_path), fourcc, fps//downsample_ratio, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"Failed to initialize video writer for clip {clip_idx}")
    
    processed_frame_count = 0
    last_frame_tracking = None
    
    try:
        # Process first frame (skip if it's a duplicate from previous clip)
        skip_first_frame = (clip_idx > 0)  # Skip duplicate frame for clips after first
        if not skip_first_frame:
            # Process first frame
            scores_dict = {}
            if hasattr(outputs, 'object_score_logits'):
                scores_dict = {
                    obj_id: outputs.object_score_logits[i]
                    for i, obj_id in enumerate(inference_session.obj_ids)
                }
            else:
                scores_dict = {obj_id: 1.0 for obj_id in inference_session.obj_ids}
            
            frame_output = {
                'frame_idx': ann_frame_idx,
                'object_ids': inference_session.obj_ids,
                'masks': {
                    obj_id: video_res_masks[i]
                    for i, obj_id in enumerate(inference_session.obj_ids)
                },
                'scores': scores_dict
            }
            
            tracked_objects = process_sam3_video_outputs(frame_output, ann_frame_idx, inference_session.obj_ids)
            
            # Calculate original video frame index accounting for downsample_ratio
            original_frame_idx = global_frame_offset + ann_frame_idx * downsample_ratio
            
            # Save outputs
            frame_output_json = {
                'frame_idx': original_frame_idx,
                'num_objects': len(tracked_objects),
                'track_ids': list(tracked_objects.keys()),
                'scores': {tid: float(obj['score']) for tid, obj in tracked_objects.items()},
                'bounding_boxes': {
                    tid: obj['box'].tolist() if obj.get('box') is not None else None
                    for tid, obj in tracked_objects.items()
                }
            }
            with open(outputs_dir / f"frame_{original_frame_idx:06d}.json", 'w') as f:
                json.dump(frame_output_json, f, indent=2)
            
            # Save masks
            for track_id, obj_info in tracked_objects.items():
                if obj_info.get('mask') is not None:
                    mask = obj_info['mask']
                    if isinstance(mask, torch.Tensor):
                        mask = mask.cpu().float().numpy() if mask.dtype == torch.bfloat16 else mask.cpu().numpy()
                    np.save(masks_dir / f"frame_{original_frame_idx:06d}_track_{track_id}.npy", mask)
            
            # Visualize and write
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, original_frame_idx)
            ret, frame = cap.read()
            cap.release()
            if ret:
                vis_frame = visualize_tracks(frame, tracked_objects, track_colors)
                out.write(vis_frame)
                processed_frame_count += 1
            
            del frame_output, tracked_objects, video_res_masks, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Process remaining frames
        for sam3_tracker_video_output in model.propagate_in_video_iterator(inference_session):
            frame_idx = sam3_tracker_video_output.frame_idx
            
            # Skip first frame if it's a duplicate
            if skip_first_frame and frame_idx == 0:
                continue
            
            video_res_masks = processor.post_process_masks(
                [sam3_tracker_video_output.pred_masks],
                original_sizes=[[inference_session.video_height, inference_session.video_width]],
            )[0]
            
            frame_output = {
                'frame_idx': frame_idx,
                'object_ids': inference_session.obj_ids,
                'masks': {
                    obj_id: video_res_masks[i]
                    for i, obj_id in enumerate(inference_session.obj_ids)
                },
                'scores': {
                    obj_id: sam3_tracker_video_output['object_score_logits'][i]
                    for i, obj_id in enumerate(inference_session.obj_ids)
                }
            }
            
            tracked_objects = process_sam3_video_outputs(frame_output, frame_idx, inference_session.obj_ids)
            
            # Save outputs
            # Calculate original video frame index accounting for frame_skip and duplicate frame
            # frame_idx is the index in the subsampled clip_frames array
            if skip_first_frame:
                # frame_idx 0 is duplicate, frame_idx 1 is first actual frame
                # Adjust frame_idx to account for duplicate, then map to original video
                actual_frame_idx = frame_idx - 1
            else:
                actual_frame_idx = frame_idx
            # Map subsampled frame index to original video frame index
            original_frame_idx = global_frame_offset + actual_frame_idx * downsample_ratio
            frame_output_json = {
                'frame_idx': original_frame_idx,
                'num_objects': len(tracked_objects),
                'track_ids': list(tracked_objects.keys()),
                'scores': {tid: float(obj['score']) for tid, obj in tracked_objects.items()},
                'bounding_boxes': {
                    tid: obj['box'].tolist() if obj.get('box') is not None else None
                    for tid, obj in tracked_objects.items()
                }
            }
            with open(outputs_dir / f"frame_{original_frame_idx:06d}.json", 'w') as f:
                json.dump(frame_output_json, f, indent=2)
            
            # Save masks
            for track_id, obj_info in tracked_objects.items():
                if obj_info.get('mask') is not None:
                    mask = obj_info['mask']
                    if isinstance(mask, torch.Tensor):
                        mask = mask.cpu().float().numpy() if mask.dtype == torch.bfloat16 else mask.cpu().numpy()
                    np.save(masks_dir / f"frame_{original_frame_idx:06d}_track_{track_id}.npy", mask)
            
            # Visualize and write - use original video frame index to read correct frame
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, original_frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                vis_frame = visualize_tracks(frame, tracked_objects, track_colors)
                out.write(vis_frame)
                processed_frame_count += 1
            
            # Store last frame tracking for next clip
            # The last frame in the inference session (excluding duplicate) is always at len(clip_frames) - 1
            # This is because if we prepended a duplicate, clip_frames has one extra frame
            # and the last actual frame is at inference frame_idx = len(clip_frames) - 1
            if frame_idx == len(clip_frames) - 1:
                last_frame_tracking = frame_output_json
            
            # Clear memory
            del frame_output, tracked_objects, video_res_masks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if (frame_idx + 1) % 50 == 0:
                print(f"  Processed frame {frame_idx + 1}/{len(clip_frames)} of clip {clip_idx + 1}")
    
    finally:
        out.release()
        # Clear clip frames and session
        del clip_frames, inference_session
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    print(f"Clip {clip_idx + 1} complete: {processed_frame_count} frames processed")
    return track_colors, last_frame_tracking, processed_frame_count


@torch.no_grad()
def process_video(
    video_path: str,
    output_dir: str,
    input_points: Optional[List[List[List[List[int]]]]] = None,
    input_labels: Optional[List[List[List[int]]]] = None,
    obj_ids: Optional[List[int]] = None,
    text_prompt: Optional[str] = None,
    num_object: Optional[int] = None,
    max_frames: Optional[int] = None,
    downsample_ratio: int = 1,
    clip_size: int = 512,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
) -> None:
    """
    Process video with SAM3 video tracking in chunks to handle long videos efficiently.
    
    The video is processed in clips of clip_size frames. For the first clip, objects are
    detected using text prompt. For subsequent clips, the last frame of the previous clip
    is prepended and tracking is initialized from the previous clip's tracking results.
    Each clip is saved as a separate video, and all clips are merged at the end.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save outputs
        input_points: List of points for each object to track (not used in chunked mode)
        input_labels: List of labels for each point (not used in chunked mode)
        obj_ids: List of object IDs (not used in chunked mode)
        text_prompt: Text prompt for first clip detection
        num_object: Maximum number of objects to track (None for all)
        max_frames: Maximum number of frames to process (None for all)
        downstample_ratio: Process every downstample_ratio frame (1 = process all frames)
        clip_size: Number of frames per clip (default: 512)
        start_time: Start time in seconds (None to start from beginning)
        end_time: End time in seconds (None to process until end)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get video information
    print("\n" + "="*60)
    print("Video Information")
    print("="*60)
    video_info = get_video_info(video_path)
    total_video_frames = video_info['total_frames']
    fps = video_info['fps']
    
    # Convert start_time and end_time to frame indices
    start_frame_idx = 0
    end_frame_idx = total_video_frames
    
    if start_time is not None:
        start_frame_idx = int(start_time * fps)
        start_frame_idx = max(0, min(start_frame_idx, total_video_frames))
    
    if end_time is not None:
        end_frame_idx = int(end_time * fps)
        end_frame_idx = max(start_frame_idx, min(end_frame_idx, total_video_frames))
    
    # Apply max_frames constraint if specified
    if max_frames is not None:
        end_frame_idx = min(end_frame_idx, start_frame_idx + max_frames)
    
    actual_max_frames = end_frame_idx - start_frame_idx
    print(f"Total frames in video: {total_video_frames}")
    print(f"Time range: {start_time if start_time is not None else 0:.2f}s - {end_time if end_time is not None else video_info['duration']:.2f}s")
    print(f"Frame range: {start_frame_idx} - {end_frame_idx}")
    print(f"Frames to process: {actual_max_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {video_info['duration']:.2f} seconds")
    print(f"Clip size: {clip_size} frames")
    
    # Calculate which clips to process
    # Clips are numbered from 0, but we need to find which clips overlap with our time range
    first_clip_idx = start_frame_idx // clip_size
    last_clip_idx = (end_frame_idx - 1) // clip_size
    num_clips_to_process = last_clip_idx - first_clip_idx + 1
    
    print(f"Clips to process: {first_clip_idx} to {last_clip_idx} ({num_clips_to_process} clips)")
    print("="*60 + "\n")
    
    # Setup device
    print("Setting up device...")
    device = Accelerator().device
    print(f"Using device: {device}\n")
    
    # Load SAM3 model and processor
    print("Loading SAM3 model and processor...")
    model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
    processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
    print("SAM3 model and processor loaded successfully!\n")
    
    # Process video in clips
    track_colors = None
    prev_clip_last_frame_tracking = None
    clip_video_paths = []
    total_processed_frames = 0
    
    # Process only clips within the time range
    for clip_idx in range(first_clip_idx, last_clip_idx + 1):
        # Calculate clip boundaries in global frame coordinates
        clip_start_frame_global = clip_idx * clip_size
        clip_end_frame_global = min(clip_start_frame_global + clip_size, total_video_frames)
        
        # Intersect with our time range
        clip_start_frame = max(clip_start_frame_global, start_frame_idx)
        clip_end_frame = min(clip_end_frame_global, end_frame_idx)
        
        # Skip if clip doesn't overlap with our range
        if clip_start_frame >= clip_end_frame:
            continue
        
        clip_frame_count = clip_end_frame - clip_start_frame
        
        print(f"\nPreparing clip {clip_idx + 1} (global frames {clip_start_frame} to {clip_end_frame-1})...")
        
        # Load frames for this clip (in global coordinates)
        clip_frames = load_video_frames_streaming(video_path, max_frames=clip_end_frame_global)
        clip_frames = clip_frames[clip_start_frame:clip_end_frame]
        # uniform sampling of the frames by a factor of frame_skip
        clip_frames = clip_frames[::downsample_ratio]
        print(f"Uniformly sampled clip frames: {len(clip_frames)}")
        print(f"Uniformly sampled clip frames shape: {clip_frames.shape}")
        
        # For clips after the first, prepend last frame from previous clip
        if clip_idx > first_clip_idx and prev_clip_last_frame_tracking is not None:
            # Load the last frame from previous clip
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start_frame - 1)
            ret, last_frame_bgr = cap.read()
            cap.release()
            
            if ret:
                last_frame_rgb = cv2.cvtColor(last_frame_bgr, cv2.COLOR_BGR2RGB)
                # Prepend to clip_frames
                clip_frames = np.concatenate([last_frame_rgb[np.newaxis, ...], clip_frames], axis=0)
                print(f"Prepended previous clip's last frame (clip now has {len(clip_frames)} frames)")
        # # save first 5 frames to check
        # for i in range(5):
        #     cv2.imwrite(f"{output_path}/clip_{clip_idx:04d}_frame_{i:04d}.jpg", clip_frames[i])
        # exit()
        # Process this clip
        track_colors, last_frame_tracking, processed_count = process_video_clip(
            clip_frames=clip_frames,
            clip_idx=clip_idx,
            global_frame_offset=clip_start_frame,
            model=model,
            processor=processor,
            device=device,
            output_path=output_path,
            video_path=video_path,
            text_prompt=text_prompt if clip_idx == first_clip_idx else None,
            num_object=num_object,
            prev_clip_last_frame_tracking=prev_clip_last_frame_tracking,
            track_colors=track_colors,
            downsample_ratio=downsample_ratio
        )
        
        # Store for next clip
        prev_clip_last_frame_tracking = last_frame_tracking
        total_processed_frames += processed_count
        
        # Store clip video path
        clips_dir = output_path / "clips"
        clip_video_paths.append(str(clips_dir / f"clip_{clip_idx:04d}.mp4"))
        
        # Clear memory
        del clip_frames
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Merge all clip videos
    print("\n" + "="*60)
    print("Merging clip videos...")
    print("="*60 + "\n")
    
    output_video_path = output_path / "tracking_visualization.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    
    merge_videos(clip_video_paths, str(output_video_path), fps // downsample_ratio)
    
    print(f"\n" + "="*60)
    print("Results saved successfully!")
    print("="*60)
    print(f"Output directory: {output_path}")
    print(f"Merged visualization video: {output_video_path}")
    print(f"Individual clip videos: {output_path / 'clips'}")
    print(f"Outputs JSON: {output_path / 'outputs'}")
    print(f"Masks: {output_path / 'masks'}")
    print(f"Total objects tracked: {len(track_colors) if track_colors else 0}")
    print(f"Total frames processed: {total_processed_frames}")
    print("="*60 + "\n")
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def main():
    """Main function to run SAM3 video tracking."""
    # Configuration
    video_path = "/data/Projects/multi-view/data/video2024-10-14T15_24_15.avi"
    video_path = '/data/Projects/multi-view/data/ibl-paw/videos/7cb81727-2097-4b52-b480-c89867b5b34c_iblrig_leftCamera.downsampled.short.mp4'
    video_path = '/data/Projects/multi-view/data/mole-rat/Aggression_1.mp4'
    video_path = '/data/Projects/multi-view/data/mole-rat/E0AAx045624_Interaction1_6-13-2025-06132025151546-0000.avi'
    video_path = '/data/Projects/multi-view/data/mole-rat/Overlaps.mp4'
    output_dir = "/data/Projects/multi-view/plots/sam3_video_tracking_outputs"
    output_dir = "/data/Projects/multi-view/plots/sam3_video_tracking_outputs_ibl-paw"
    output_dir = "/data/Projects/multi-view/plots/sam3_video_tracking_outputs_mole-rat_aggression"
    output_dir = "/data/Projects/multi-view/plots/sam3_video_tracking_outputs_mole-rat_interaction"
    output_dir = "/data/Projects/multi-view/plots/sam3_video_tracking_outputs_mole-rat_overlaps"
    max_frames = None  # Set to None to process all frames, or a number to limit
    downsample_ratio = 1
    start_time = 30
    end_time = 10000
    
    # Option 1: Use text prompt to detect objects automatically
    # text_prompt = "fishes"  # Text prompt for detection
    # text_prompt = "mouse tongue"
    text_prompt = "mouse"
    # num_object = 3  # Track top 3 objects (None to track all)
    num_object = 2
    
    print(f"Using SAM3 Video Tracking API")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA available: {torch.cuda.is_available()}\n")
    print(f'video path: {video_path}')
    # Process video
    process_video(
        video_path=video_path,
        output_dir=output_dir,
        text_prompt=text_prompt,
        num_object=num_object,
        max_frames=max_frames,
        downsample_ratio=downsample_ratio,
        start_time=start_time,  # Set to start time in seconds (None for beginning)
        end_time=end_time     # Set to end time in seconds (None for end)
    )


if __name__ == "__main__":
    main()


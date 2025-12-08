"""
SAM3 Multi-Animal Tracking with Text Prompts

This script uses SAM3 to track multiple animals in videos using text prompts.
It segments all instances matching the text prompt in each frame and tracks
them across frames using IoU-based association.
"""

import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from matplotlib.colors import hsv_to_rgb
from transformers import Sam3Processor, Sam3Model
import tempfile
import gc
import os


class MultiObjectTracker:
    """Simple IoU-based tracker for maintaining object identities across frames."""
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5, lock_tracks: bool = False):
        """
        Args:
            iou_threshold: Minimum IoU to associate detections with tracks
            max_age: Maximum frames a track can be missing before deletion
            lock_tracks: If True, don't create new tracks after initialization
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: Dict[int, Dict] = {}  # track_id -> track info
        self.next_id = 1
        self.lock_tracks = lock_tracks  # Prevent creating new tracks after initialization
        
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes in xyxy format."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, boxes: np.ndarray, scores: np.ndarray, masks: Optional[np.ndarray] = None) -> Dict[int, Dict]:
        """
        Update tracker with new detections.
        
        Args:
            boxes: Array of shape (N, 4) in xyxy format
            scores: Array of shape (N,) with confidence scores
            masks: Optional array of shape (N, H, W) with binary masks
            
        Returns:
            Dictionary mapping track_id to detection info
        """
        # Remove old tracks
        active_tracks = {}
        for track_id, track in self.tracks.items():
            track['age'] += 1
            if track['age'] <= self.max_age:
                active_tracks[track_id] = track
        self.tracks = active_tracks
        
        if len(boxes) == 0:
            return {}
        
        # Calculate IoU matrix between detections and existing tracks
        n_detections = len(boxes)
        n_tracks = len(self.tracks)
        
        if n_tracks == 0:
            # Initialize new tracks for all detections
            matches = {}
            for i in range(n_detections):
                track_id = self.next_id
                self.next_id += 1
                matches[track_id] = i
        else:
            # Build IoU matrix
            iou_matrix = np.zeros((n_tracks, n_detections))
            track_ids = list(self.tracks.keys())
            
            for t_idx, track_id in enumerate(track_ids):
                for d_idx in range(n_detections):
                    iou_matrix[t_idx, d_idx] = self._calculate_iou(
                        self.tracks[track_id]['box'], boxes[d_idx]
                    )
            
            # Greedy matching
            matches = {}
            used_detections = set()
            
            # Sort by IoU (highest first)
            match_candidates = []
            for t_idx, track_id in enumerate(track_ids):
                for d_idx in range(n_detections):
                    if iou_matrix[t_idx, d_idx] >= self.iou_threshold:
                        match_candidates.append((iou_matrix[t_idx, d_idx], track_id, d_idx))
            
            match_candidates.sort(reverse=True)
            
            for iou, track_id, d_idx in match_candidates:
                if track_id not in matches and d_idx not in used_detections:
                    matches[track_id] = d_idx
                    used_detections.add(d_idx)
            
            # Create new tracks for unmatched detections (only if not locked)
            if not self.lock_tracks:
                for d_idx in range(n_detections):
                    if d_idx not in used_detections:
                        track_id = self.next_id
                        self.next_id += 1
                        matches[track_id] = d_idx
        
        # Update tracks
        tracked_objects = {}
        for track_id, det_idx in matches.items():
            box = boxes[det_idx]
            score = scores[det_idx]
            mask = masks[det_idx] if masks is not None else None
            
            if track_id in self.tracks:
                # Update existing track
                self.tracks[track_id]['box'] = box
                self.tracks[track_id]['score'] = score
                self.tracks[track_id]['mask'] = mask
                self.tracks[track_id]['age'] = 0
            else:
                # Create new track
                self.tracks[track_id] = {
                    'box': box,
                    'score': score,
                    'mask': mask,
                    'age': 0
                }
            
            tracked_objects[track_id] = {
                'box': box,
                'score': score,
                'mask': mask
            }
        
        return tracked_objects


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


def load_sam3_model(device: str = 'cuda') -> Tuple[Sam3Model, Sam3Processor]:
    """
    Load SAM3 model and processor.
    
    Args:
        device: Device to run inference on ('cuda' or 'cpu')
        
    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading SAM3 model on {device}...")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model.eval()
    print("SAM3 model loaded successfully!")
    return model, processor


def segment_frame(
    model: Sam3Model,
    processor: Sam3Processor,
    frame: np.ndarray,
    text_prompt: str,
    device: str,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segment objects in a frame using text prompt.
    
    Args:
        model: SAM3 model
        processor: SAM3 processor
        frame: Frame as numpy array (H, W, 3) in BGR format
        text_prompt: Text prompt for segmentation
        device: Device to run inference on
        threshold: Confidence threshold for detections
        
    Returns:
        Tuple of (masks, boxes, scores)
        - masks: Binary masks array (N, H, W)
        - boxes: Bounding boxes in xyxy format (N, 4)
        - scores: Confidence scores (N,)
    """
    # Convert BGR to RGB and to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Process with SAM3
    inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process results
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    
    masks = results['masks'].cpu().numpy()  # (N, H, W)
    boxes = results['boxes'].cpu().numpy()  # (N, 4) in xyxy format
    scores = results['scores'].cpu().numpy()  # (N,)
    
    return masks, boxes, scores


def generate_track_colors(n_tracks: int) -> np.ndarray:
    """
    Generate distinct colors for each track.
    
    Args:
        n_tracks: Number of tracks
        
    Returns:
        Array of RGB colors (N, 3) with values in [0, 255]
    """
    colors = []
    for i in range(n_tracks):
        hue = i / max(n_tracks, 1)
        color = hsv_to_rgb([hue, 0.8, 0.9])
        colors.append([int(c * 255) for c in color])
    return np.array(colors)


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
        if obj_info['mask'] is not None:
            mask = obj_info['mask']
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                mask = mask.astype(np.uint8)
            
            # Create colored overlay
            overlay = vis_frame.copy()
            overlay[mask > 0] = color_bgr
            vis_frame = cv2.addWeighted(vis_frame, 0.6, overlay, 0.4, 0)
        
        # Draw bounding box
        box = obj_info['box'].astype(int)
        x1, y1, x2, y2 = box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color_bgr, 2)
        
        # Draw track ID and score
        label = f"ID:{track_id} ({obj_info['score']:.2f})"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y = max(y1, label_size[1] + 10)
        
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


def process_video(
    video_path: str,
    output_path: str,
    text_prompt: str,
    device: str = 'cuda',
    threshold: float = 0.5,
    tracker_iou_threshold: float = 0.3,
    max_frames: Optional[int] = None,
    num_object: Optional[int] = None,
    use_temp_file: bool = True,
    clear_cache_interval: int = 50,
    frame_skip: int = 0
) -> None:
    """
    Process video with SAM3 tracking and save visualization.
    
    Memory and I/O optimized for long videos with thousands of frames.
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video
        text_prompt: Text prompt for segmentation
        device: Device to run inference on
        threshold: Confidence threshold for detections
        tracker_iou_threshold: IoU threshold for track association
        max_frames: Maximum number of frames to process (None for all frames)
        num_object: Maximum number of top objects to track (None for all detected objects)
        use_temp_file: If True, write to temp file first then move to final location (safer for long videos)
        clear_cache_interval: Clear GPU cache every N frames (0 to disable)
        frame_skip: Process every (frame_skip+1) frame (0 = process all frames)
    """
    # Get video information
    print("\n" + "="*60)
    print("Video Information")
    print("="*60)
    video_info = get_video_info(video_path)
    print(f"Total frames: {video_info['total_frames']}")
    print(f"FPS: {video_info['fps']:.2f}")
    print(f"Duration: {video_info['duration']:.2f} seconds")
    print("="*60 + "\n")
    
    # Load SAM3 model
    model, processor = load_sam3_model(device)
    
    # Initialize tracker with lock_tracks if num_object is specified
    # This ensures we only track the objects selected in the first frame
    lock_tracks = num_object is not None
    tracker = MultiObjectTracker(iou_threshold=tracker_iou_threshold, lock_tracks=lock_tracks)
    track_colors = {}
    first_frame_processed = False
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer with optimized codec and temp file if needed
    final_output_path = output_path
    if use_temp_file:
        # Use temp file to avoid corruption if process is interrupted
        temp_dir = Path(output_path).parent
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.mp4', 
            dir=str(temp_dir), 
            delete=False
        )
        temp_path = temp_file.name
        temp_file.close()
        output_path = temp_path
        print(f"Writing to temporary file: {temp_path}")
    
    # Use H.264 codec for better compression and compatibility
    # Try different codecs in order of preference
    codecs_to_try = [
        ('avc1', 'H.264/AVC'),  # Best compression
        ('mp4v', 'MPEG-4'),      # Fallback
        ('XVID', 'Xvid'),        # Another fallback
    ]
    
    out = None
    for fourcc_str, codec_name in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            print(f"Using video codec: {codec_name} ({fourcc_str})")
            break
        else:
            if out is not None:
                out.release()
            out = None
    
    if out is None or not out.isOpened():
        raise RuntimeError("Failed to initialize video writer with any codec")
    
    frame_idx = 0
    processed_frame_count = 0
    frames_to_process = min(max_frames, video_info['total_frames']) if max_frames is not None else video_info['total_frames']
    print(f"Processing video with text prompt: '{text_prompt}'")
    if frame_skip > 0:
        print(f"Frame skip: {frame_skip} (processing every {frame_skip+1} frame)")
    print(f"Processing {frames_to_process} frames (out of {video_info['total_frames']} total)...\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if we've reached max_frames
            if max_frames is not None and processed_frame_count >= max_frames:
                break
            
            # Frame skipping for faster processing
            if frame_skip > 0 and frame_idx % (frame_skip + 1) != 0:
                frame_idx += 1
                continue
            
            # Segment frame
            masks, boxes, scores = segment_frame(
                model, processor, frame, text_prompt, device, threshold
            )
            
            # For first frame: select top num_object objects and initialize tracker
            # For subsequent frames: track only those initial objects
            if not first_frame_processed:
                # First frame: select top num_object objects
                if num_object is not None and len(scores) > 0:
                    # Sort by scores in descending order and take top N
                    top_indices = np.argsort(scores)[::-1][:num_object]
                    masks = masks[top_indices]
                    boxes = boxes[top_indices]
                    scores = scores[top_indices]
                    print(f"First frame: Selected top {len(scores)} objects to track throughout video")
                first_frame_processed = True
            # For subsequent frames, don't filter - let tracker match to existing tracks only
            
            # Update tracker (will only match to existing tracks if lock_tracks=True)
            tracked_objects = tracker.update(boxes, scores, masks)
            
            # Update track colors for new tracks
            for track_id in tracked_objects.keys():
                if track_id not in track_colors:
                    # Generate color based on track ID
                    hue = (track_id * 0.618) % 1.0  # Golden ratio for better distribution
                    color = hsv_to_rgb([hue, 0.8, 0.9])
                    track_colors[track_id] = np.array([int(c * 255) for c in color])
            
            # Visualize
            vis_frame = visualize_tracks(frame, tracked_objects, track_colors)
            
            # Write frame
            out.write(vis_frame)
            
            # Memory management: clear intermediate variables
            del masks, boxes, scores, vis_frame
            if device.startswith('cuda'):
                # Clear GPU cache periodically
                if clear_cache_interval > 0 and processed_frame_count % clear_cache_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            processed_frame_count += 1
            frame_idx += 1
            
            if processed_frame_count % 10 == 0:
                print(f"Processed {processed_frame_count}/{frames_to_process} frames "
                      f"({len(tracked_objects)} objects tracked)")
    
    finally:
        cap.release()
        out.release()
        
        # Move temp file to final location if using temp file
        if use_temp_file and os.path.exists(output_path):
            if os.path.exists(final_output_path):
                os.remove(final_output_path)
            os.rename(output_path, final_output_path)
            print(f"\nVideo saved to: {final_output_path}")
        else:
            print(f"\nVideo saved to: {output_path}")
        
        print(f"Total objects tracked: {len(tracker.tracks)}")
        
        # Final cleanup
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        gc.collect()


def main():
    """Main function to run SAM3 tracking."""
    # Configuration
    video_path = "/data/Projects/multi-view/data/video2024-10-14T15_24_15.avi"
    output_path = "/data/Projects/multi-view/plots/sam3_tracking_output.mp4"
    text_prompt = "fishes"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    threshold = 0.5  # Confidence threshold for detections
    tracker_iou_threshold = 0.3  # IoU threshold for track association
    
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA available: {torch.cuda.is_available()}\n")
    device = 'cuda:5'
    max_frames = None  # Set to None to process all frames, or a number to limit
    # max_frames = 256
    num_object = 3  # Set to desired number (e.g., 5) to track top N objects, None for all
    
    # Memory and I/O optimization settings for long videos
    use_temp_file = True  # Write to temp file first (safer for long videos)
    clear_cache_interval = 0  # Clear GPU cache every N frames (0 to disable)
    frame_skip = 0  # Process every (frame_skip+1) frame (0 = process all frames)
    
    # Process video
    process_video(
        video_path=video_path,
        output_path=output_path,
        text_prompt=text_prompt,
        device=device,
        threshold=threshold,
        tracker_iou_threshold=tracker_iou_threshold,
        max_frames=max_frames,
        num_object=num_object,
        use_temp_file=use_temp_file,
        clear_cache_interval=clear_cache_interval,
        frame_skip=frame_skip
    )


if __name__ == "__main__":
    main()


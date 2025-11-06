import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from data.sbm_ds import decode_rle_segmentation
from utils.metric_utils import (compute_bbox_iou, compute_mask_iou,
                                compute_precision_recall_f1, mask_to_bbox)
from utils.viz_utils import (add_text_label, create_video_from_frames,
                             create_visualization_video, generate_colors,
                             overlay_multiple_masks_and_bboxes,
                             stack_frames_vertically)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_predictions(predictions_file: Path) -> Dict:
    """Load predictions from JSON file.
    
    Args:
        predictions_file: Path to predictions.json
    
    Returns:
        Dictionary with 'predictions' and 'metadata' keys
    """
    with open(predictions_file, 'r') as f:
        data = json.load(f)
    return data


def load_ground_truth(annotations_file: Path) -> Dict:
    """Load ground truth annotations from JSON file.
    
    Args:
        annotations_file: Path to annotations JSON file
    
    Returns:
        Dictionary with annotations data
    """
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    return data


def match_predictions_to_gt(
    predictions: List[Dict],
    gt_annotations: List[Dict]
) -> Dict[Tuple[int, int, str], Dict]:
    """Match predictions to ground truth based on ann_id, video_id, and frame_path.
    
    Args:
        predictions: List of prediction dictionaries
        gt_annotations: List of ground truth annotation dictionaries
    
    Returns:
        Dictionary mapping (ann_id, video_id, frame_idx) -> matched data
    """
    # Index ground truth by (video_id, ann_id)
    gt_index = {}
    for ann in gt_annotations:
        video_id = ann['video_id']
        ann_id = ann['id']
        key = (video_id, ann_id)
        gt_index[key] = ann
    
    # Match predictions to ground truth
    matched_data = {}
    
    for pred in predictions:
        ann_id = pred['ann_id']
        video_id = pred['video_id']
        frame_path = pred['frame_path']
        
        # Extract frame index from path (e.g., "1/00000.jpg" -> 0)
        frame_name = Path(frame_path).name
        try:
            frame_idx = int(frame_name.replace('.jpg', ''))
        except ValueError:
            logger.warning(f"Could not parse frame index from {frame_path}")
            continue
        
        key = (video_id, ann_id)
        if key not in gt_index:
            logger.warning(f"No ground truth found for video_id={video_id}, ann_id={ann_id}")
            continue
        
        gt_ann = gt_index[key]
        if frame_idx >= len(gt_ann['segmentations']):
            logger.warning(
                f"Frame index {frame_idx} out of range for video_id={video_id}, "
                f"ann_id={ann_id} (length={len(gt_ann['segmentations'])})"
            )
            continue
        
        match_key = (ann_id, video_id, frame_idx)
        matched_data[match_key] = {
            'prediction': pred,
            'ground_truth': gt_ann,
            'frame_idx': frame_idx
        }
    
    return matched_data


def evaluate_predictions(
    predictions_file: Path,
    gt_annotations_file: Path,
    return_matched_data: bool = False
) -> Tuple[Dict, Optional[Dict]]:
    """Evaluate predictions against ground truth.
    
    Args:
        predictions_file: Path to predictions.json
        gt_annotations_file: Path to ground truth annotations JSON
    
    Returns:
        Dictionary with evaluation metrics, and optionally matched_data dict
        if return_matched_data=True
    """
    logger.info(f"Loading predictions from {predictions_file}")
    pred_data = load_predictions(predictions_file)
    predictions = pred_data['predictions']
    metadata = pred_data.get('metadata', {})
    
    logger.info(f"Loading ground truth from {gt_annotations_file}")
    gt_data = load_ground_truth(gt_annotations_file)
    gt_annotations = gt_data['annotations']
    
    logger.info(f"Found {len(predictions)} predictions and {len(gt_annotations)} annotations")
    
    # Match predictions to ground truth
    matched_data = match_predictions_to_gt(predictions, gt_annotations)
    logger.info(f"Matched {len(matched_data)} predictions to ground truth")
    
    # Initialize metric accumulators
    iou_scores = []
    bbox_iou_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    per_annotation_metrics = defaultdict(list)
    
    # Evaluate each matched prediction
    for (ann_id, video_id, frame_idx), match_info in matched_data.items():
        pred = match_info['prediction']
        gt_ann = match_info['ground_truth']
        
        # Decode masks
        try:
            # Decode predicted mask
            pred_rle = pred['pred_rle_mask']
            pred_mask = decode_rle_segmentation(pred_rle, return_tensor=False)
            
            # Decode ground truth mask
            gt_rle = gt_ann['segmentations'][frame_idx]
            gt_mask = decode_rle_segmentation(gt_rle, return_tensor=False)
            
            # Ensure masks are binary
            pred_mask = (pred_mask > 0).astype(np.uint8)
            gt_mask = (gt_mask > 0).astype(np.uint8)
            
        except Exception as e:
            logger.warning(
                f"Failed to decode masks for video_id={video_id}, ann_id={ann_id}, "
                f"frame_idx={frame_idx}: {e}"
            )
            continue
        
        # Compute mask metrics
        iou = compute_mask_iou(pred_mask, gt_mask)
        iou_scores.append(iou)
        per_annotation_metrics[ann_id].append(iou)
        
        # Compute precision, recall, F1
        prf = compute_precision_recall_f1(pred_mask, gt_mask)
        precision_scores.append(prf['precision'])
        recall_scores.append(prf['recall'])
        f1_scores.append(prf['f1'])
        
        # Compute bbox from masks
        pred_bbox = mask_to_bbox(pred_mask)
        gt_bbox = gt_ann['bboxes'][frame_idx]
        
        # Compute bbox IoU
        bbox_iou = compute_bbox_iou(pred_bbox, gt_bbox)
        bbox_iou_scores.append(bbox_iou)
        
        logger.debug(
            f"video_id={video_id}, ann_id={ann_id}, frame_idx={frame_idx}: "
            f"IoU={iou:.4f}, bbox_IoU={bbox_iou:.4f}, F1={prf['f1']:.4f}"
        )
    
    # Aggregate metrics
    if len(iou_scores) == 0:
        logger.warning("No predictions were successfully evaluated!")
        metrics = {
            'mask_iou': {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0},
            'mask_miou': 0.0,
            'bbox_iou': {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0},
            'precision': {'mean': 0.0, 'std': 0.0},
            'recall': {'mean': 0.0, 'std': 0.0},
            'f1': {'mean': 0.0, 'std': 0.0},
            'per_annotation_miou': {},
            'num_evaluated': 0,
            'num_predictions': len(predictions),
            'num_annotations': len(gt_annotations)
        }
    else:
        metrics = {
            'mask_iou': {
                'mean': float(np.mean(iou_scores)),
                'std': float(np.std(iou_scores)),
                'median': float(np.median(iou_scores)),
                'min': float(np.min(iou_scores)),
                'max': float(np.max(iou_scores))
            },
            'mask_miou': float(np.mean(iou_scores)),  # Mean IoU
            'bbox_iou': {
                'mean': float(np.mean(bbox_iou_scores)),
                'std': float(np.std(bbox_iou_scores)),
                'median': float(np.median(bbox_iou_scores)),
                'min': float(np.min(bbox_iou_scores)),
                'max': float(np.max(bbox_iou_scores))
            },
            'precision': {
                'mean': float(np.mean(precision_scores)),
                'std': float(np.std(precision_scores))
            },
            'recall': {
                'mean': float(np.mean(recall_scores)),
                'std': float(np.std(recall_scores))
            },
            'f1': {
                'mean': float(np.mean(f1_scores)),
                'std': float(np.std(f1_scores))
            },
            'per_annotation_miou': {
                ann_id: float(np.mean(scores))
                for ann_id, scores in per_annotation_metrics.items()
            },
            'num_evaluated': len(iou_scores),
            'num_predictions': len(predictions),
            'num_annotations': len(gt_annotations)
        }
    
    if return_matched_data:
        return metrics, matched_data
    return metrics, None


def create_visualization_videos(
    matched_data: Dict[Tuple[int, int, int], Dict],
    predictions_file: Path,
    gt_annotations_file: Path,
    output_dir: Path,
    fps: int = 10,
    mask_alpha: float = 0.5
) -> None:
    """Create visualization videos for predicted and ground truth masks/bboxes.
    
    Merges multiple annotations on the same frame into a single video.
    
    Args:
        matched_data: Dictionary mapping (ann_id, video_id, frame_idx) -> match_info
        predictions_file: Path to predictions.json (for reference)
        gt_annotations_file: Path to ground truth annotations (for reference)
        output_dir: Directory to save videos
        fps: Frames per second for videos
        mask_alpha: Transparency of mask overlay
    """
    from PIL import Image
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions file to get model name from metadata
    pred_data = load_predictions(predictions_file)
    metadata = pred_data.get('metadata', {})
    model_name = metadata.get('model_name', 'Unknown Model')
    
    # Group matched data by (video_id, frame_idx) to merge annotations on same frame
    video_frames = defaultdict(lambda: defaultdict(list))
    for (ann_id, video_id, frame_idx), match_info in matched_data.items():
        video_frames[video_id][frame_idx].append((ann_id, match_info))
    
    # Generate colors for different annotations
    unique_ann_ids = sorted(set(ann_id for (ann_id, _, _) in matched_data.keys()))
    colors = generate_colors(len(unique_ann_ids))
    ann_id_to_color = {ann_id: colors[i] for i, ann_id in enumerate(unique_ann_ids)}
    
    logger.info(f"Creating visualization videos for {len(video_frames)} videos...")
    
    # Process each video
    for video_id, frame_dict in video_frames.items():
        # Sort frames by frame index
        sorted_frames = sorted(frame_dict.items())
        
        # Collect all frames with all annotations
        pred_visualized_frames = []
        gt_visualized_frames = []
        frame_indices = []  # Track frame indices for labeling
        
        for frame_idx, ann_data_list in sorted_frames:
            # Get frame path from first annotation (all annotations on same frame share same path)
            frame_path = ann_data_list[0][1]['prediction']['frame_path']
            
            # Load base image
            try:
                img = Image.open(frame_path).convert('RGB')
                img_array = np.array(img)
            except Exception as e:
                logger.warning(f"Failed to load frame {frame_path}: {e}")
                continue
            
            # Collect all masks and bboxes for this frame
            pred_masks = []
            pred_bboxes = []
            pred_colors = []
            pred_labels = []
            
            gt_masks = []
            gt_bboxes = []
            gt_colors = []
            gt_labels = []
            
            for ann_id, match_info in ann_data_list:
                pred = match_info['prediction']
                gt_ann = match_info['ground_truth']
                
                color = ann_id_to_color[ann_id]
                
                try:
                    # Decode predicted mask
                    pred_rle = pred['pred_rle_mask']
                    pred_mask = decode_rle_segmentation(pred_rle, return_tensor=False)
                    pred_mask = (pred_mask > 0).astype(np.uint8)
                    
                    # Decode ground truth mask
                    gt_rle = gt_ann['segmentations'][frame_idx]
                    gt_mask = decode_rle_segmentation(gt_rle, return_tensor=False)
                    gt_mask = (gt_mask > 0).astype(np.uint8)
                    
                    # Get bboxes
                    pred_bbox = mask_to_bbox(pred_mask)
                    gt_bbox = gt_ann['bboxes'][frame_idx]
                    
                    pred_masks.append(pred_mask)
                    pred_bboxes.append(pred_bbox.tolist() if isinstance(pred_bbox, np.ndarray) else pred_bbox)
                    pred_colors.append(color)
                    pred_labels.append(f'Pred {ann_id}')
                    
                    gt_masks.append(gt_mask)
                    gt_bboxes.append(gt_bbox)
                    gt_colors.append(color)
                    gt_labels.append(f'GT {ann_id}')
                    
                except Exception as e:
                    logger.warning(
                        f"Failed to process annotation for video_id={video_id}, "
                        f"ann_id={ann_id}, frame_idx={frame_idx}: {e}"
                    )
                    continue
            
            # Overlay all masks and bboxes on the same frame
            if pred_masks:
                pred_frame = overlay_multiple_masks_and_bboxes(
                    img_array,
                    pred_masks,
                    pred_bboxes,
                    pred_colors,
                    mask_alpha=mask_alpha,
                    bbox_thickness=2,
                    labels=pred_labels
                )
                pred_visualized_frames.append(pred_frame)
                frame_indices.append(frame_idx)  # Store frame index
            
            if gt_masks:
                gt_frame = overlay_multiple_masks_and_bboxes(
                    img_array,
                    gt_masks,
                    gt_bboxes,
                    gt_colors,
                    mask_alpha=mask_alpha,
                    bbox_thickness=2,
                    labels=gt_labels
                )
                gt_visualized_frames.append(gt_frame)
        
        if len(pred_visualized_frames) == 0:
            logger.warning(f"No valid frames for video_id={video_id}")
            continue
        
        # Ensure same number of frames
        num_frames = min(len(pred_visualized_frames), len(gt_visualized_frames))
        pred_visualized_frames = pred_visualized_frames[:num_frames]
        gt_visualized_frames = gt_visualized_frames[:num_frames]
        frame_indices = frame_indices[:num_frames]  # Trim frame_indices to match
        
        # Add text labels to frames
        pred_labeled_frames = []
        gt_labeled_frames = []
        
        for idx, (pred_frame, gt_frame) in enumerate(zip(pred_visualized_frames, gt_visualized_frames)):
            # Get frame number
            frame_num = frame_indices[idx] if idx < len(frame_indices) else idx
            
            # Add "Prediction" label with model name and frame number to top frame
            prediction_text = f'Prediction: {model_name} | Frame num: {frame_num}'
            pred_labeled = add_text_label(
                pred_frame,
                prediction_text,
                position='top-left',
                font_scale=0.6,
                thickness=2,
                bg_color=(0, 0, 255),  # Red background
                text_color=(255, 255, 255)  # White text
            )
            pred_labeled_frames.append(pred_labeled)
            
            # Add "Ground Truth" label to bottom frame
            gt_labeled = add_text_label(
                gt_frame,
                'Ground Truth',
                position='top-left',
                font_scale=0.6,
                thickness=2,
                bg_color=(0, 255, 0),  # Green background
                text_color=(255, 255, 255)  # White text
            )
            gt_labeled_frames.append(gt_labeled)
        
        # Stack frames vertically (prediction on top, GT on bottom)
        merged_frames = []
        for pred_frame, gt_frame in zip(pred_labeled_frames, gt_labeled_frames):
            merged_frame = stack_frames_vertically(pred_frame, gt_frame)
            merged_frames.append(merged_frame)
        
        # Create merged video
        merged_video_path = output_dir / f'video_{video_id}_merged.mp4'
        logger.info(f"Creating merged video (Prediction top, GT bottom): {merged_video_path}")
        create_video_from_frames(merged_frames, merged_video_path, fps=fps)
    
    logger.info(f"Visualization videos created in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate SBM dataset predictions')
    parser.add_argument(
        '--predictions_file',
        type=str,
        required=True,
        help='Path to predictions.json file'
    )
    parser.add_argument(
        '--gt_annotations_file',
        type=str,
        required=True,
        help='Path to ground truth annotations JSON file'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Path to save evaluation results JSON (default: predictions_dir/evaluation_results.json)'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Set the logging level'
    )
    parser.add_argument(
        '--output_videos',
        type=str,
        default=None,
        help='Directory to save visualization videos (default: predictions_dir/videos). '
             'If specified, creates videos with predicted and ground truth masks/bboxes'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=1,
        help='Frames per second for output videos'
    )
    parser.add_argument(
        '--mask_alpha',
        type=float,
        default=0.5,
        help='Transparency of mask overlay (0.0 to 1.0)'
    )
    args = parser.parse_args()
    
    # Configure logging
    log_level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(log_level)
    
    predictions_file = Path(args.predictions_file)
    gt_annotations_file = Path(args.gt_annotations_file)
    
    if not predictions_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
    if not gt_annotations_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_annotations_file}")
    
    # Evaluate predictions
    logger.info("Starting evaluation...")
    return_matched = args.output_videos is not None
    metrics, matched_data = evaluate_predictions(
        predictions_file, 
        gt_annotations_file,
        return_matched_data=return_matched
    )
    
    # Save results
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = predictions_file.parent / 'evaluation_results.json'
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Evaluation results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Number of predictions evaluated: {metrics['num_evaluated']}")
    print(f"\nMask Metrics:")
    print(f"  Mean IoU (mIoU): {metrics['mask_miou']:.4f}")
    print(f"  IoU - Mean: {metrics['mask_iou']['mean']:.4f} ± {metrics['mask_iou']['std']:.4f}")
    print(f"  IoU - Median: {metrics['mask_iou']['median']:.4f}")
    print(f"  IoU - Range: [{metrics['mask_iou']['min']:.4f}, {metrics['mask_iou']['max']:.4f}]")
    print(f"\nBounding Box Metrics:")
    print(f"  Bbox IoU - Mean: {metrics['bbox_iou']['mean']:.4f} ± {metrics['bbox_iou']['std']:.4f}")
    print(f"  Bbox IoU - Median: {metrics['bbox_iou']['median']:.4f}")
    print(f"\nPrecision/Recall/F1:")
    print(f"  Precision: {metrics['precision']['mean']:.4f} ± {metrics['precision']['std']:.4f}")
    print(f"  Recall: {metrics['recall']['mean']:.4f} ± {metrics['recall']['std']:.4f}")
    print(f"  F1 Score: {metrics['f1']['mean']:.4f} ± {metrics['f1']['std']:.4f}")
    print("="*60)

    # Generate visualization videos if requested
    if args.output_videos is not None and matched_data is not None:
        logger.info("Generating visualization videos...")
        video_output_dir = Path(args.output_videos) if args.output_videos else predictions_file.parent / 'videos'
        
        create_visualization_videos(
            matched_data=matched_data,
            predictions_file=predictions_file,
            gt_annotations_file=gt_annotations_file,
            output_dir=video_output_dir,
            fps=args.fps,
            mask_alpha=args.mask_alpha
        )
        logger.info(f"Videos saved to: {video_output_dir}") 


if __name__ == '__main__':
    main()


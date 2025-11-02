import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from data.sbm_ds import (SBMDataset, decode_rle_segmentation,
                         encode_mask_to_rle, mask_centroid)

# Get module logger (will be configured in main())
logger = logging.getLogger(__name__)


def build_sam(model_name: str, device: str):
    """Build SAM model and processor from Hugging Face."""
    from transformers import SamModel, SamProcessor
    processor = SamProcessor.from_pretrained(model_name)
    model = SamModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model


@torch.no_grad()
def run_sam_with_point_prompt(
    image: Image.Image,
    point: tuple,
    processor,
    model,
    device: str
) -> np.ndarray:
    """Run SAM inference with a single central point prompt.
    
    Args:
        image: PIL Image
        point: (x, y) center point coordinate
        processor: SAM processor
        model: SAM model
        device: Device to run on
    
    Returns:
        Binary mask as numpy array (H, W) of uint8
    """
    # Format point as required by SAM: [[x, y]] with label 1 (foreground)
    # SAM expects input_points as list of lists: [[x, y]] for single point
    input_points = [[point]]  # Single point per image
    input_labels = [[1]]  # 1 = foreground point
    
    inputs = processor(
        images=image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
              for k, v in inputs.items()}
    outputs = model(**inputs, multimask_output=False)
    
    # Post-process masks
    masks = processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"]
    )
    # Extract single mask and convert to binary
    # masks is a list, first element is for first image
    # shape: (num_masks, H, W) -> we take first mask
    mask = masks[0].squeeze(0).detach().cpu().numpy()  # Remove batch dim
    if mask.ndim > 2:
        mask = mask[0]  # Take first mask if multiple
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    return binary_mask


def main():
    parser = argparse.ArgumentParser(description='SBM dataset SAM inference with central point prompts')
    parser.add_argument(
        '--data_root',
        type=str,
        default='/work/hdd/beez/ywang74/Project/multi-view/data/SBeA_dataset/SM_fig1_data/SBM-VIS-VIStR-12',
        help='Path to SBM-VIS-VIStR-12 directory'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'valid', 'test'],
        help='Dataset split to use'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='facebook/sam-vit-base',
        help='Hugging Face model name for SAM'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run inference on'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Directory to save predicted masks (default: data_root/outputs/sbm_sam_masks/{split})'
    )
    parser.add_argument(
        '--resize',
        type=int,
        nargs=2,
        default=None,
        help='Optional resize (width height) for images'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Set the logging level'
    )
    args = parser.parse_args()
    
    # Configure logging with the specified level
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True  # Override any existing configuration (Python 3.8+)
    )
    # Ensure both root logger and module logger are set to the correct level
    logging.root.setLevel(log_level)
    logger.setLevel(log_level)
    
    # Also set handler level if it exists
    for handler in logging.root.handlers:
        handler.setLevel(log_level)
    
    # Setup save directory
    if args.save_dir is None:
        data_root = Path(args.data_root)
        args.save_dir = data_root.parent / 'outputs' / 'sbm_sam_masks' / args.split
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting SBM inference pipeline")
    logger.info(f"Dataset root: {args.data_root}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Save directory: {save_dir}")
    logger.info(f"Device: {args.device}")
    
    # Create dataset
    dataset = SBMDataset(
        data_root=args.data_root,
        split=args.split,
        resize=tuple(args.resize) if args.resize else None
    )
    
    logger.info(f"Loaded dataset: {len(dataset)} clips, split={args.split}")
    
    # Build SAM model
    logger.info(f"Loading SAM model: {args.model_name}")
    processor, model = build_sam(args.model_name, args.device)
    logger.info(f"SAM model loaded successfully on {args.device}")
    
    # Collect prediction results for JSON export
    prediction_results = []
    
    # Run inference - iterate directly over dataset to avoid collation issues
    for clip_idx in range(len(dataset)):
        batch = dataset[clip_idx]  # Dictionary with preprocessed data
        
        # Unpack batch (already preprocessed in dataset)
        clip_images = batch['clip']  # List of PIL Images
        frame_paths = batch['frame_paths']  # List of paths
        video_id = batch['video_id']  # int
        annotations = batch['annotations']  # List of annotation dicts
        
        # Get video dimensions
        orig_width = batch.get('width')
        orig_height = batch.get('height')
        
        clip_info = {
            'width': orig_width,
            'height': orig_height,
        }
        
        logger.info(
            f"Processing clip {clip_idx+1}/{len(dataset)}: video_id={video_id}, "
            f"{len(clip_images)} frames, {len(annotations)} annotations"
        )
        
        # Process each annotation instance separately
        for ann_idx, ann in enumerate(annotations):
            ann_id = ann['id']
            gt_segmentations = ann['segmentations']
            assert len(clip_images) == len(gt_segmentations), f"Number of clip images {len(clip_images)} does not match number of segmentations {len(gt_segmentations)}"
            # Process each frame
            for frame_idx, img in enumerate(clip_images):
                # Decode segmentation mask for this frame
                gt_mask = decode_rle_segmentation(gt_segmentations[frame_idx], width=img.width, height=img.height)
                # Calculate centroid from segmentation mask (in original size)
                center_point = mask_centroid(gt_mask)
                # Run SAM inference with central point prompt
                pred_mask = run_sam_with_point_prompt(
                    img,
                    center_point,
                    processor,
                    model,
                    args.device
                )
                
                # Encode mask to RLE format for storage/evaluation
                pred_rle_mask = encode_mask_to_rle(pred_mask, return_dict=True)
                logger.debug(f"Encoded mask to RLE: {pred_rle_mask['size']}, counts length: {len(pred_rle_mask['counts'])}")
                
                # Store prediction result for JSON export
                frame_path = frame_paths[frame_idx]
                prediction_result = {
                    'ann_id': ann_id,
                    'video_id': video_id,
                    'frame_path': frame_path,
                    'pred_rle_mask': {
                        'counts': pred_rle_mask['counts'],
                        'size': pred_rle_mask['size']
                    },
                    'prompt': {
                        'type': 'point',
                        'coordinates': [float(center_point[0]), float(center_point[1])]
                    }
                }
                prediction_results.append(prediction_result)
    
    # Save prediction results to JSON file with metadata
    json_output_file = save_dir / 'predictions.json'
    output_data = {
        'predictions': prediction_results,
        'metadata': {
            'model_name': args.model_name,
            'data_root': args.data_root,
            'split': args.split,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'author': 'Yanchen'
        }
    }
    
    with open(json_output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Inference complete! Masks saved to: {save_dir}")
    logger.info(f"Prediction results saved to: {json_output_file} ({len(prediction_results)} predictions)")


if __name__ == '__main__':
    """
    Example usage:
    
    # Run inference on test split
    python src/inference_sbm.py \
        --data_root /work/hdd/beez/ywang74/Project/multi-view/data/SBeA_dataset/SM_fig1_data/SBM-VIS-VIStR-12 \
        --split test \
        --model_name facebook/sam-vit-base \
        --device cuda
    
    # Run inference on valid split with custom save directory and debug logging
    python src/inference_sbm.py \
        --data_root /work/hdd/beez/ywang74/Project/multi-view/data/SBeA_dataset/SM_fig1_data/SBM-VIS-VIStR-12 \
        --split valid \
        --save_dir /path/to/output/masks \
        --log_level DEBUG
    """
    main()


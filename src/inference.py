import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader

from data.crim13_track_ds import Crim13TrackDataset


def build_sam(model_name: str, device: str):
    from transformers import SamModel, SamProcessor
    processor = SamProcessor.from_pretrained(model_name)
    model = SamModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model


def _load_labels_ood(labels_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv, header=[0, 1, 2], index_col=0)
    return df


def _median_keypoints(df_row: pd.Series) -> Dict[str, dict]:
    """Extract keypoints for both mice, returning {'black_mouse': {...}, 'white_mouse': {...}}"""
    s = df_row.copy()
    s.index = pd.MultiIndex.from_tuples(s.index)
    median = s.groupby(level=[1, 2]).median(numeric_only=True)
    
    mice_kps = {'black_mouse': {}, 'white_mouse': {}}
    for bodypart in sorted(set([idx[0] for idx in median.index])):
        try:
            x = float(median[(bodypart, 'x')])
            y = float(median[(bodypart, 'y')])
            if 'black_mouse' in bodypart:
                # Extract keypoint name without mouse prefix
                kp_name = bodypart.replace('black_mouse_', '')
                mice_kps['black_mouse'][kp_name] = (x, y)
            elif 'white_mouse' in bodypart:
                kp_name = bodypart.replace('white_mouse_', '')
                mice_kps['white_mouse'][kp_name] = (x, y)
        except Exception:
            continue
    return mice_kps


def _keypoints_to_box(kps: dict, pad: float = 10.0) -> Tuple[float, float, float, float]:
    """Convert keypoints dict to bounding box (x0, y0, x1, y1)."""
    if len(kps) == 0:
        return (0.0, 0.0, 1.0, 1.0)
    xs = np.array([pt[0] for pt in kps.values()], dtype=float)
    ys = np.array([pt[1] for pt in kps.values()], dtype=float)
    x0, y0 = float(xs.min()), float(ys.min())
    x1, y1 = float(xs.max()), float(ys.max())
    return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)


def _create_pseudo_gt_mask(kps: dict, img_h: int, img_w: int, radius: int = 5) -> np.ndarray:
    """Create a pseudo-GT mask from keypoints by placing small circular regions around each keypoint."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if len(kps) == 0:
        return mask
    
    y_coords, x_coords = np.ogrid[:img_h, :img_w]
    for pt in kps.values():
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < img_w and 0 <= y < img_h:
            # Create circular region around keypoint
            dist_sq = (x_coords - x) ** 2 + (y_coords - y) ** 2
            mask[dist_sq <= radius ** 2] = 1
    return mask


def _compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def _compute_keypoint_coverage(mask: np.ndarray, kps: dict, img_w: int, img_h: int) -> float:
    """Compute fraction of keypoints that fall inside the mask."""
    if len(kps) == 0:
        return np.nan
    xs = np.clip(np.array([pt[0] for pt in kps.values()], dtype=int), 0, img_w - 1)
    ys = np.clip(np.array([pt[1] for pt in kps.values()], dtype=int), 0, img_h - 1)
    inside = mask[ys, xs] > 0
    return float(inside.mean())


def _compute_keypoint_to_mask_distance(mask: np.ndarray, kps: dict, img_w: int, img_h: int) -> float:
    """Compute average distance from keypoints to nearest mask boundary."""
    if len(kps) == 0:
        return np.nan
    
    # Find mask boundary
    from scipy.ndimage import binary_erosion
    mask_eroded = binary_erosion(mask)
    boundary = mask.astype(bool) & ~mask_eroded
    
    # Get boundary coordinates
    boundary_y, boundary_x = np.where(boundary)
    if len(boundary_x) == 0:
        return np.nan
    
    boundary_points = np.column_stack([boundary_x, boundary_y])
    
    # Compute distances from each keypoint to nearest boundary point
    distances = []
    for pt in kps.values():
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < img_w and 0 <= y < img_h:
            kp_point = np.array([[x, y]])
            dists = cdist(kp_point, boundary_points)
            distances.append(float(dists.min()))
    
    return np.mean(distances) if distances else np.nan


def _check_mouse_separation(mask1: np.ndarray, mask2: np.ndarray, kps1: dict, kps2: dict, 
                           img_w: int, img_h: int) -> Dict[str, float]:
    """Check if keypoints for each mouse fall in separate connected components."""
    from scipy.ndimage import label

    # Combine both masks
    combined_mask = ((mask1 > 0) | (mask2 > 0)).astype(int)
    labeled, num_components = label(combined_mask)
    
    # Find which component each keypoint belongs to
    components1 = []
    components2 = []
    
    for pt in kps1.values():
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < img_w and 0 <= y < img_h and combined_mask[y, x] > 0:
            components1.append(int(labeled[y, x]))
    
    for pt in kps2.values():
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < img_w and 0 <= y < img_h and combined_mask[y, x] > 0:
            components2.append(int(labeled[y, x]))
    
    # Check separation: if all keypoints from each mouse are in different components
    unique1 = set(components1) if components1 else set()
    unique2 = set(components2) if components2 else set()
    
    separated = len(unique1 & unique2) == 0 and len(unique1) > 0 and len(unique2) > 0
    overlap_ratio = len(unique1 & unique2) / max(len(unique1 | unique2), 1) if (unique1 | unique2) else 0.0
    
    return {
        'separated': float(separated),
        'component_overlap_ratio': overlap_ratio,
        'num_components': num_components,
    }


@torch.no_grad()
def run_sam_masks(image: Image.Image, boxes_xyxy: List[List[float]], processor, model, device: str) -> List[np.ndarray]:
    inputs = processor(images=image, input_boxes=[boxes_xyxy], return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    outputs = model(**inputs, multimask_output=False)
    masks = processor.post_process_masks(
        outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
    )
    out = masks[0].squeeze(1).detach().cpu().numpy()  # (num_prompts, H, W)
    return [out[i] for i in range(out.shape[0])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=str(Path(__file__).resolve().parents[1] / 'data' / 'crim13_track'))
    parser.add_argument('--model_name', type=str, default='facebook/sam-vit-base')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch over clips; keep 1 if variable lengths')
    parser.add_argument('--save_dir', type=str, default=str(Path(__file__).resolve().parents[1] / 'data' / 'outputs' / 'crim13_sam_masks'))
    args = parser.parse_args()

    dataset = Crim13TrackDataset(data_root=args.data_root, resize=None)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    processor, model = build_sam(args.model_name, args.device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    labels_csv = Path(args.data_root) / 'labels_OOD.csv'
    labels_df = _load_labels_ood(labels_csv)
    csv_dir = labels_csv.parent

    for batch in loader:
        # batch_size is 1 by default; unpack
        # Each field is a list length B; we handle B>1 by iterating
        clips = batch['clip']  # (B, T, C, H, W) if batch_size>1 and T aligned; otherwise a list-like
        sessions = batch['session']
        start_idxs = batch['start_idx']
        end_idxs = batch['end_idx']
        frame_paths_batch = batch['frame_paths']

        # Normalize to iterable over items
        if isinstance(sessions, list):
            indices = range(len(sessions))
        else:
            # Default collate turns strings into list only if B>1; when B==1, they are scalars
            indices = [0]
            # Wrap scalars
            sessions = [sessions]
            start_idxs = [start_idxs]
            end_idxs = [end_idxs]
            frame_paths_batch = [frame_paths_batch]

        for bi in indices:
            session = sessions[bi]
            start_i = int(start_idxs[bi])
            end_i = int(end_idxs[bi])
            frame_paths = frame_paths_batch[bi]

            clip_masks_black = []
            clip_masks_white = []
            frame_metrics = []
            
            for fp in frame_paths:
                img = Image.open(fp).convert('RGB')
                w, h = img.size
                print(f'Image size: {w}x{h}')

                rel_path = str(Path(fp).relative_to(csv_dir))
                assert rel_path in labels_df.index, f'{rel_path} not in labels_df.index'

                row = labels_df.loc[rel_path]
                mice_kps = _median_keypoints(row)
                black_kps = mice_kps.get('black_mouse', {})
                white_kps = mice_kps.get('white_mouse', {})

                # Segment each mouse separately using their respective bounding boxes
                masks_black, masks_white = None, None
                if len(black_kps) > 0:
                    box_black = list(_keypoints_to_box(black_kps, pad=10.0))
                    box_black[0] = float(max(0, min(box_black[0], w - 1)))
                    box_black[1] = float(max(0, min(box_black[1], h - 1)))
                    box_black[2] = float(max(0, min(box_black[2], w - 1)))
                    box_black[3] = float(max(0, min(box_black[3], h - 1)))
                    masks_black = run_sam_masks(img, [box_black], processor, model, args.device)
                    mask_black = (masks_black[0] > 0.5).astype(np.uint8)
                else:
                    mask_black = np.zeros((h, w), dtype=np.uint8)

                if len(white_kps) > 0:
                    box_white = list(_keypoints_to_box(white_kps, pad=10.0))
                    box_white[0] = float(max(0, min(box_white[0], w - 1)))
                    box_white[1] = float(max(0, min(box_white[1], h - 1)))
                    box_white[2] = float(max(0, min(box_white[2], w - 1)))
                    box_white[3] = float(max(0, min(box_white[3], h - 1)))
                    masks_white = run_sam_masks(img, [box_white], processor, model, args.device)
                    mask_white = (masks_white[0] > 0.5).astype(np.uint8)
                else:
                    mask_white = np.zeros((h, w), dtype=np.uint8)

                clip_masks_black.append(mask_black)
                clip_masks_white.append(mask_white)

                # Compute metrics for quantitative evaluation
                metrics = {}
                
                # Black mouse metrics
                if len(black_kps) > 0:
                    pseudo_gt_black = _create_pseudo_gt_mask(black_kps, h, w, radius=5)
                    metrics['black_mouse'] = {
                        'keypoint_coverage': _compute_keypoint_coverage(mask_black, black_kps, w, h),
                        'pseudo_gt_iou': _compute_iou(mask_black, pseudo_gt_black),
                        'keypoint_to_mask_distance': _compute_keypoint_to_mask_distance(mask_black, black_kps, w, h),
                    }
                else:
                    metrics['black_mouse'] = {
                        'keypoint_coverage': np.nan,
                        'pseudo_gt_iou': np.nan,
                        'keypoint_to_mask_distance': np.nan,
                    }
                
                # White mouse metrics
                if len(white_kps) > 0:
                    pseudo_gt_white = _create_pseudo_gt_mask(white_kps, h, w, radius=5)
                    metrics['white_mouse'] = {
                        'keypoint_coverage': _compute_keypoint_coverage(mask_white, white_kps, w, h),
                        'pseudo_gt_iou': _compute_iou(mask_white, pseudo_gt_white),
                        'keypoint_to_mask_distance': _compute_keypoint_to_mask_distance(mask_white, white_kps, w, h),
                    }
                else:
                    metrics['white_mouse'] = {
                        'keypoint_coverage': np.nan,
                        'pseudo_gt_iou': np.nan,
                        'keypoint_to_mask_distance': np.nan,
                    }
                
                # Mouse separation metrics
                if len(black_kps) > 0 and len(white_kps) > 0:
                    separation_metrics = _check_mouse_separation(mask_black, mask_white, black_kps, white_kps, w, h)
                    metrics['separation'] = separation_metrics
                else:
                    metrics['separation'] = {'separated': np.nan, 'component_overlap_ratio': np.nan}
                
                frame_metrics.append(metrics)

            # Compute temporal consistency metrics (mask area stability across frames)
            mask_areas_black = [m.sum() for m in clip_masks_black]
            mask_areas_white = [m.sum() for m in clip_masks_white]
            temporal_consistency = {
                'black_mouse_area_mean': float(np.mean(mask_areas_black)),
                'black_mouse_area_std': float(np.std(mask_areas_black)),
                'white_mouse_area_mean': float(np.mean(mask_areas_white)),
                'white_mouse_area_std': float(np.std(mask_areas_white)),
            }
            
            # Aggregate metrics across frames
            summary_metrics = {
                'black_mouse_keypoint_coverage_mean': np.nanmean([m['black_mouse']['keypoint_coverage'] for m in frame_metrics]),
                'black_mouse_pseudo_gt_iou_mean': np.nanmean([m['black_mouse']['pseudo_gt_iou'] for m in frame_metrics]),
                'white_mouse_keypoint_coverage_mean': np.nanmean([m['white_mouse']['keypoint_coverage'] for m in frame_metrics]),
                'white_mouse_pseudo_gt_iou_mean': np.nanmean([m['white_mouse']['pseudo_gt_iou'] for m in frame_metrics]),
                'mouse_separation_rate': np.nanmean([m['separation']['separated'] for m in frame_metrics]),
            }
            
            out_path = save_dir / f'{session}_clip_{start_i:06d}_{end_i:06d}.pt'
            torch.save({
                'masks_black': clip_masks_black,
                'masks_white': clip_masks_white,
                'session': session,
                'start_idx': start_i,
                'end_idx': end_i,
                'frame_paths': frame_paths,
                'frame_metrics': frame_metrics,  # Per-frame detailed metrics
                'summary_metrics': summary_metrics,  # Aggregated metrics
                'temporal_consistency': temporal_consistency,
            }, out_path)
            print(f'Saved {len(clip_masks_black)} frame masks (black & white) -> {out_path}')
            print(f'  Summary: black_mouse_kp_coverage={summary_metrics["black_mouse_keypoint_coverage_mean"]:.3f}, '
                  f'white_mouse_kp_coverage={summary_metrics["white_mouse_keypoint_coverage_mean"]:.3f}, '
                  f'separation_rate={summary_metrics["mouse_separation_rate"]:.3f}')


if __name__ == '__main__':
    main()



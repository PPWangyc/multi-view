import argparse
import deeplabcut
import numpy as np
import os
import pandas as pd
import yaml
import cv2
import matplotlib.cm as cm
from datetime import datetime

# Import DLC utilities
from deeplabcut.utils import auxiliaryfunctions


# Paths - update these to match your workspace
WORKSPACE_ROOT = '/data/Projects/multi-view'
dlc_dir = os.path.join(WORKSPACE_ROOT, 'dlc_projects')
ptl_dir = os.path.join(WORKSPACE_ROOT, 'data')

displayiters = 500
saveiters = 5000
maxiters = 50000


# Dataset configuration
DATASET_CONFIG = {
    'mirror-mouse': {
        'scorer': 'rick',
        'date': '2025-11-24',  # Use current date
        'date_str': 'Nov24',  # Auto-generate
        'global_scale': 0.64,
        'total_frames': 789,
        'train_frames_configs': {
            75: {'shuffles': [750, 751, 752, 753, 754], 'trainingsetindex': 0, 'trainingset': 49},
            100: {'shuffles': [1000, 1001, 1002, 1003, 1004], 'trainingsetindex': 0, 'trainingset': 13},
            'default': {'shuffles': [10, 11, 12, 13, 14], 'trainingsetindex': 1, 'trainingset': 89}
        }
    },
    'mirror-fish': {
        'scorer': 'rick',
        'date': '2025-11-24',
        'date_str': 'Nov24',
        'global_scale': 0.7,
        'total_frames': None,
        'train_frames_configs': {
            100: {'shuffles': [1000, 1001, 1002, 1003, 1004], 'trainingsetindex': 0, 'trainingset': 95},
        }
    },
    'ibl-paw': {
        'scorer': 'lightning_tracker',
        'date': '2025-11-24',
        'date_str': 'Nov24',
        'global_scale': 1.28,
        'total_frames': None,
        'train_frames_configs': {
            100: {'shuffles': [1000, 1001, 1002, 1003, 1004], 'trainingsetindex': 0, 'trainingset': 13},
        }
    },
    'crim13': {
        'scorer': 'mt',
        'date': '2025-11-24',  # Use current date
        'date_str': 'Nov24',  # Auto-generate
        'global_scale': 0.53,
        'total_frames': 3986,
        'train_frames_configs': {
            100: {'shuffles': [1000, 1001, 1002, 1003, 1004], 'trainingsetindex': 0, 'trainingset': 95},
        }
    }
}

def get_dataset_config(dataset_name, train_frames):
    """Get configuration for a dataset and train_frames combination."""
    if dataset_name not in DATASET_CONFIG:
        raise NotImplementedError(f"Dataset '{dataset_name}' not configured")
    
    config = DATASET_CONFIG[dataset_name].copy()
    
    # Handle date and date_str
    if config['date'] is None:
        config['date'] = datetime.now().strftime('%Y-%m-%d')
    if config['date_str'] is None:
        config['date_str'] = datetime.now().strftime('%b%d').replace('0', '').title()
    
    # Get train_frames specific config
    train_config = config['train_frames_configs'].get(
        train_frames, 
        config['train_frames_configs'].get('default')
    )
    
    config.update(train_config)
    return config

def get_existing_shuffles(training_datasets_dir):
    """Extract existing shuffle numbers from training dataset pickle files."""
    if not os.path.exists(training_datasets_dir):
        return []
    
    dataset_dirs = [d for d in os.listdir(training_datasets_dir) 
                   if os.path.isdir(os.path.join(training_datasets_dir, d))]
    if not dataset_dirs:
        return []
    
    dataset_dir = os.path.join(training_datasets_dir, dataset_dirs[0])
    pickle_files = [f for f in os.listdir(dataset_dir) if 'pickle' in f]
    
    existing_shuffles = []
    for pf in pickle_files:
        parts = pf.split('shuffle')
        if len(parts) > 1:
            try:
                shuffle_num = int(parts[1].split('.')[0])
                existing_shuffles.append(shuffle_num)
            except ValueError:
                continue
    
    return sorted(existing_shuffles)

def run_main(args):    
    # Get dataset configuration
    config = get_dataset_config(args.dataset, args.train_frames)
    scorer = config['scorer']
    date = config['date']
    date_str = config['date_str']
    global_scale = config['global_scale']
    shuffle_list = config['shuffles']
    trainingsetindex = config['trainingsetindex']
    trainingset = config['trainingset']
    total_frames = config.get('total_frames')

    # Setup project paths
    project_dir = os.path.join(dlc_dir, '%s-%s-%s' % (args.dataset, scorer, date))
    config_path = os.path.join(project_dir, 'config.yaml')
    
    # Initialize training_params (will be populated from config if available)
    training_params = {}
    batch_size = 8
    epochs = 300
    if args.dataset == 'crim13':
        learning_rate = 0.001
    elif args.dataset == 'ibl-paw':
        learning_rate = 0.0001
    elif args.dataset == 'mirror-mouse':
        learning_rate = 0.0001
    elif args.dataset == 'mirror-fish':
        learning_rate = 0.0005
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")
    gamma = 0.5
    milestones = [150, 200, 250]
    learning_rate_list = [learning_rate * gamma, learning_rate * gamma ** 2, learning_rate * gamma ** 3]
    print(f"config path: {config_path}")

    # Load config and validate training dataset
    if os.path.exists(config_path):
        cfg = auxiliaryfunctions.read_config(config_path)
        training_fractions = cfg.get('TrainingFraction', [])
        training_datasets_dir = os.path.join(cfg['project_path'], 'training-datasets', 'iteration-0')
        # Read training parameters from config if available (set by setup script)
        training_params = cfg.get('training_params', {})
        if training_params:
            batch_size = training_params.get('batch_size', batch_size)
            print(f"Using batch size from config: {batch_size}")
            # Note: learning_rate and max_epochs are stored but DLC uses iterations, not epochs
            # maxiters is set globally, but we could use max_epochs to calculate it if needed
        
        # Verify existing shuffles match requested ones
        existing_shuffles = get_existing_shuffles(training_datasets_dir)
        if existing_shuffles:
            print(f"Found {len(existing_shuffles)} existing shuffles in training dataset: {existing_shuffles}")
            missing_shuffles = [s for s in shuffle_list if s not in existing_shuffles]
            if missing_shuffles:
                print(f"Warning: Shuffles {missing_shuffles} not found in training dataset")
                shuffle_list = [s for s in shuffle_list if s in existing_shuffles]
                print(f"Using only existing shuffles: {shuffle_list}")
        
        # Update training fraction index from config if available
        # For crim13 with train_frames=100, we should use index 0 (which has 0.95)
        # Don't override if trainingsetindex is already set correctly
        if training_fractions:
            # For train_frames=100, always use index 0 (first training fraction, typically 0.95)
            if args.train_frames == 100:
                trainingsetindex = 0
                trainingset = int(round(training_fractions[0] * 100))
                print(f"Using training fraction {training_fractions[0]:.2%} (index {trainingsetindex}) for train_frames=100")
    else:
        print(f"Warning: Config file not found at {config_path}")
        print("Using default training set index 0")
        if trainingset is None:
            trainingset = int(round((args.train_frames / (total_frames or 1000)) * 100)) if total_frames else 0
            print(f"Warning: Set trainingset to {trainingset} as fallback (config not found)")
    
    # Final safety check
    if trainingset is None:
        trainingset = 0
        print(f"Warning: trainingset was None, setting to 0 as final fallback")
    
    print(f"Training with shuffles: {shuffle_list}")
    for shuffle in shuffle_list:
        # Use PyTorch model directory (DLC 3.0 default)
        model_folder = os.path.join(
            project_dir, 'dlc-models-pytorch', 'iteration-0', '%s%s-trainset%ishuffle%i' % (
                args.dataset, date_str, trainingset, shuffle,
            ))

        # Update PyTorch config files (only if they exist)
        # Note: Config files are created during training, so they might not exist yet
        pytorch_config_file = os.path.join(model_folder, 'train', 'pytorch_config.yaml')
        print(f"Updating PyTorch config: {pytorch_config_file}")
        pytorch_config = yaml.safe_load(open(pytorch_config_file))
        # update batch size
        pytorch_config['train_settings']['batch_size'] = batch_size
        # update epochs
        pytorch_config['train_settings']['epochs'] = epochs 
        # update optimizer
        pytorch_config['runner']['optimizer']['type'] = 'Adam'
        # update learning rate
        # pytorch_config['runner']['optimizer']['params']['lr'] = float(training_params['learning_rate'])
        pytorch_config['runner']['optimizer']['params']['lr'] = learning_rate
        # update eval interval
        pytorch_config['runner']['eval_interval'] = 5
        # update scheduler
        pytorch_config['runner']['scheduler']['params']['lr_list'] = [[lr] for lr in learning_rate_list]
        pytorch_config['runner']['scheduler']['params']['milestones'] = milestones
        pytorch_config['runner']['scheduler']['type'] = 'LRListScheduler'
        yaml.dump(pytorch_config, open(pytorch_config_file, 'w'))
        print(f"Updated PyTorch config: {pytorch_config_file}")
        # Update test config if it exists
        test_config_file = os.path.join(model_folder, 'test', 'pose_cfg.yaml')
        test_config = yaml.safe_load(open(test_config_file))
        test_config['global_scale'] = global_scale
        yaml.dump(test_config, open(test_config_file, 'w'))

        # train model
        if not args.skip_train:
            deeplabcut.train_network(
                config_path,
                shuffle=shuffle,
                trainingsetindex=trainingsetindex,
                gputouse=args.gpu_id,
                max_snapshots_to_keep=1,
                autotune=False,
                displayiters=displayiters,
                saveiters=saveiters,
                maxiters=maxiters,
                allow_growth=True,
            )
        
        # Run inference on OOD labeled data from CollectedData_new.csv
        if args.dataset == 'crim13':
            ood_csv_name = 'labels_new.csv'
        elif args.dataset == 'mirror-mouse':
            ood_csv_name = 'CollectedData_new.csv'
        elif args.dataset == 'mirror-fish':
            ood_csv_name = 'CollectedData_new.csv'
        elif args.dataset == 'ibl-paw':
            ood_csv_name = 'CollectedData_new.csv'
        else:
            raise ValueError(f"Dataset {args.dataset} not supported.")
        ood_csv_file = os.path.join(ptl_dir, args.dataset, ood_csv_name)
        print(f"Running inference on OOD labeled data: {ood_csv_file}")
        # Read the OOD CSV to get image paths
        df_ood = pd.read_csv(ood_csv_file, header=[0, 1, 2], index_col=0)
        image_paths = df_ood.index.tolist()
        
        # Get bodypart names from the CSV
        bodyparts = df_ood.columns.get_level_values('bodyparts').unique().tolist()
        
        print(f"Found {len(image_paths)} images to process")
        print(f"Bodyparts: {bodyparts}")
        
        # Resolve image paths relative to the data directory
        project_path = cfg['project_path']
        data_dir = os.path.join(ptl_dir, args.dataset)
        full_image_paths = []
        missing_images = []
        
        for img_path in image_paths:
            # Handle paths like "labeled-data_OOD/180607_004/img015278.png"
            full_image_paths.append(os.path.join(data_dir, img_path))
        
        print(f"Processing {len(full_image_paths)} images...")
        
        # Use DLC's analyze_images function for PyTorch models
        print(f"Running DLC inference on {len(full_image_paths)} images...")
        
        # Run inference using analyze_images
        # Note: analyze_images uses 'device' parameter, not 'gputouse'
        
        dlc_inference = deeplabcut.analyze_images(
            config_path,
            full_image_paths,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            device='cuda',
            save_as_csv=True,
        )
        
        # DLC saves predictions in the same directory as the images
        # Find the prediction CSV files created by DLC
        prediction_files = []
        
        # Check in the directories where images are located
        image_dirs = set(os.path.dirname(img_path) for img_path in full_image_paths)
        for img_dir in image_dirs:
            if os.path.exists(img_dir):
                for f in os.listdir(img_dir):
                    if f.endswith('.csv') and 'predictions' in f.lower() and 'image_predictions' in f.lower():
                        prediction_files.append(os.path.join(img_dir, f))
                    elif f.endswith('.h5'):
                        # remove the h5 file
                        os.remove(os.path.join(img_dir, f))
        # Create output directory in the DLC project folder: {project_path}/ood_results/predictions_new_shuffle{shuffle}/
        ood_results_dir = os.path.join(project_path, 'ood_results')
        shuffle_dir = os.path.join(ood_results_dir, f'predictions_new_shuffle{shuffle}')
        os.makedirs(shuffle_dir, exist_ok=True)
        
        # Try to load and reformat the predictions
        # DLC may save predictions in different formats
        # We need to create predictions_new.csv with the same structure as CollectedData_new.csv
        predictions_file = os.path.join(shuffle_dir, 'predictions_new.csv')
        
        # DLC typically saves predictions with scorer name in the filename
        # Find the file that matches the current shuffle
        pred_file = None
        for pf in prediction_files:
            if f'shuffle{shuffle}' in pf:
                pred_file = pf
                break
        print(f"Loading prediction file: {pred_file}")
        # Read the prediction file
        # DLC saves with MultiIndex columns
        df_pred = pd.read_csv(pred_file, header=[0, 1, 2], index_col=0)
        # remove the predictions_file
        for pf in prediction_files:
            os.remove(pf)
        # Filter out any non-image rows (like 'coords' header rows that might have been read as data)
        header_keywords = ['coords', 'bodyparts', 'scorer']
        valid_mask = ~df_pred.index.astype(str).str.lower().isin(header_keywords)
        df_pred = df_pred[valid_mask]
        
        # Ensure the index matches the original image paths from CollectedData_new.csv
        # Map full paths back to relative paths
        path_mapping = {full: orig for full, orig in zip(full_image_paths, image_paths)}
        
        # Update index to match original paths
        new_index = []
        for idx in df_pred.index:
            if idx in path_mapping:
                new_index.append(path_mapping[idx])
            elif os.path.basename(idx) in [os.path.basename(p) for p in image_paths]:
                # Try to match by filename
                matching_path = [p for p in image_paths if os.path.basename(p) == os.path.basename(idx)]
                if matching_path:
                    new_index.append(matching_path[0])
                else:
                    new_index.append(idx)
            else:
                new_index.append(idx)
        
        df_pred.index = new_index
        
        # Convert DLC format to expected format: (scorer, bodyparts, coords)
        # DLC format: (scorer, individuals, bodyparts) where bodyparts have .1, .2 suffixes
        # Expected format: (scorer, bodyparts, coords) where coords are 'x', 'y', 'likelihood'
        
        # Build new columns in the expected format
        new_columns = []
        new_data = {}
        
        # Get scorer name (use first one found)
        scorer_name = df_pred.columns.get_level_values(0)[0] if df_pred.columns.nlevels > 0 else 'DLC_scorer'
        
        # Process each column to extract bodypart and coordinate
        for col in df_pred.columns:
            if isinstance(col, tuple):
                if len(col) == 3:
                    # (scorer, individuals, bodypart)
                    scorer, individuals, bodypart = col
                elif len(col) == 4:
                    # (scorer, individuals, bodypart, coord) - already has coords
                    scorer, individuals, bodypart, coord = col
                    base_bodypart = bodypart.split('.')[0] if '.' in str(bodypart) else bodypart
                    new_col = (scorer_name, base_bodypart, coord)
                    if new_col not in new_columns:
                        new_columns.append(new_col)
                        new_data[new_col] = df_pred[col]
                    continue
                else:
                    continue
                
                # Infer coordinate from bodypart suffix
                bodypart_str = str(bodypart)
                base_bodypart = bodypart.split('.')[0] if '.' in bodypart_str else bodypart
                
                if bodypart_str.endswith('.1'):
                    # This is the y coordinate
                    coord = 'y'
                elif bodypart_str.endswith('.2'):
                    # This is likelihood
                    coord = 'likelihood'
                else:
                    # This is the x coordinate (base name)
                    coord = 'x'
                
                new_col = (scorer_name, base_bodypart, coord)
                if new_col not in new_columns:
                    new_columns.append(new_col)
                    new_data[new_col] = df_pred[col]
        
        # Create new DataFrame with expected structure
        if new_columns:
            # Sort columns: group by bodypart, then by coord (x, y, likelihood)
            bodyparts = sorted(set(col[1] for col in new_columns))
            coord_order = ['x', 'y', 'likelihood']
            sorted_columns = []
            for bp in bodyparts:
                for coord in coord_order:
                    col = (scorer_name, bp, coord)
                    if col in new_columns:
                        sorted_columns.append(col)
            
            # Build DataFrame with sorted columns
            df_pred_formatted = pd.DataFrame(
                {col: new_data[col] for col in sorted_columns},
                index=df_pred.index
            )
            df_pred_formatted.columns = pd.MultiIndex.from_tuples(
                sorted_columns,
                names=['scorer', 'bodyparts', 'coords']
            )
        else:
            # Fallback: try to drop individuals level if present
            if df_pred.columns.nlevels > 3:
                df_pred_formatted = df_pred.droplevel('individuals', axis=1, errors='ignore')
            else:
                df_pred_formatted = df_pred.copy()
        
        # Reorder columns to match ground truth order for consistency
        # This ensures the prediction DataFrame has the same column order as GT
        try:
            # Load GT to get the correct order
            df_gt_for_order = pd.read_csv(ood_csv_file, header=[0, 1, 2], index_col=0)
            gt_bodyparts_order = df_gt_for_order.columns.get_level_values('bodyparts').unique().tolist()
            gt_scorer_order = df_gt_for_order.columns.get_level_values('scorer')[0]
            
            # Build columns in GT order
            coord_order = ['x', 'y', 'likelihood']
            reordered_columns = []
            for gt_bp in gt_bodyparts_order:
                # Check if this bodypart exists in predictions
                for coord in coord_order:
                    col = (scorer_name, gt_bp, coord)
                    if col in df_pred_formatted.columns:
                        reordered_columns.append(col)
            
            # Also add any prediction bodyparts not in GT (shouldn't happen, but be safe)
            existing_cols = set(reordered_columns)
            for col in df_pred_formatted.columns:
                if col not in existing_cols:
                    reordered_columns.append(col)
            
            # Reorder the DataFrame
            if reordered_columns:
                df_pred_formatted = df_pred_formatted[reordered_columns]
                print(f"  Reordered columns to match GT order")
        except Exception as e:
            print(f"  Warning: Could not reorder columns to match GT: {e}")
            print(f"  Using alphabetical order instead")
        
        # Save as predictions_new.csv in the DLC project's ood_results/ folder
        df_pred_formatted.to_csv(predictions_file)
        print(f"✓ Predictions saved to: {predictions_file}")
        print(f"  Shape: {df_pred_formatted.shape}")
        print(f"  Columns structure: {df_pred_formatted.columns.nlevels} levels")
        print(f"  Column names: {df_pred_formatted.columns.names}")
        print(f"  Index sample: {df_pred_formatted.index[:3].tolist()}")
        
        # Update df_pred for pixel error computation
        df_pred = df_pred_formatted
        
        # Overlay keypoints on images
        print("\nOverlaying keypoints on images...")
        try:
            overlay_images_dir = shuffle_dir
            images_overlaid = 0
            images_failed = 0
            
            # Get bodyparts and their colors
            bodyparts = sorted(set(col[1] for col in df_pred_formatted.columns if len(col) == 3))
            # Create a color map for bodyparts
            colors = cm.get_cmap('tab20')(np.linspace(0, 1, len(bodyparts)))
            
            # Use the path mapping that was already created
            # path_mapping maps full paths to original relative paths
            # We need the reverse: original paths to full paths
            orig_to_full_path = {orig: full for full, orig in path_mapping.items()}
            
            for img_idx, img_path in enumerate(df_pred_formatted.index):
                if img_idx % 50 == 0:
                    print(f"  Processing image {img_idx+1}/{len(df_pred_formatted)}...")
                
                # Skip header rows
                if isinstance(img_path, str) and img_path.lower() in ['coords', 'bodyparts', 'scorer']:
                    continue
                
                # Find the full image path using the mapping
                full_img_path = orig_to_full_path.get(img_path)
                
                # If not in mapping, try to find it
                if not full_img_path or not os.path.exists(full_img_path):
                    if img_path.startswith('labeled-data_OOD/'):
                        rel_path = img_path.replace('labeled-data_OOD/', '')
                        full_img_path = os.path.join(data_dir, 'labeled-data_OOD', rel_path)
                        if not os.path.exists(full_img_path):
                            full_img_path = os.path.join(project_path, 'labeled-data_OOD', rel_path)
                    else:
                        if os.path.isabs(img_path):
                            full_img_path = img_path if os.path.exists(img_path) else None
                        else:
                            full_img_path = os.path.join(project_path, img_path)
                            if not os.path.exists(full_img_path):
                                full_img_path = os.path.join(data_dir, img_path)
                
                if full_img_path and os.path.exists(full_img_path):
                    try:
                        # Load image
                        img = cv2.imread(full_img_path)
                        if img is None:
                            images_failed += 1
                            continue
                        
                        # Draw keypoints
                        for bp_idx, bodypart in enumerate(bodyparts):
                            # Get x, y coordinates for this bodypart
                            x_col = (scorer_name, bodypart, 'x')
                            y_col = (scorer_name, bodypart, 'y')
                            likelihood_col = (scorer_name, bodypart, 'likelihood')
                            
                            if x_col in df_pred_formatted.columns and y_col in df_pred_formatted.columns:
                                try:
                                    x = df_pred_formatted.loc[img_path, x_col]
                                    y = df_pred_formatted.loc[img_path, y_col]
                                    likelihood = df_pred_formatted.loc[img_path, likelihood_col] if likelihood_col in df_pred_formatted.columns else 1.0
                                    
                                    # Only draw if coordinates are valid and likelihood is reasonable
                                    if pd.notna(x) and pd.notna(y) and pd.notna(likelihood):
                                        x, y = int(float(x)), int(float(y))
                                        likelihood = float(likelihood)
                                        
                                        # Skip if likelihood is too low
                                        if likelihood < 0.1:
                                            continue
                                        
                                        # Get color for this bodypart (convert from matplotlib to BGR)
                                        color_rgba = colors[bp_idx % len(colors)]
                                        color_bgr = (int(color_rgba[2] * 255), int(color_rgba[1] * 255), int(color_rgba[0] * 255))
                                        
                                        # Draw circle for keypoint
                                        radius = max(3, int(5 * likelihood))  # Size based on likelihood
                                        cv2.circle(img, (x, y), radius, color_bgr, -1)
                                        # Draw smaller circle outline
                                        cv2.circle(img, (x, y), radius + 1, (255, 255, 255), 1)
                                        
                                        # Optionally add bodypart label
                                        # cv2.putText(img, bodypart, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_bgr, 1)
                                except (KeyError, ValueError, TypeError) as e:
                                    continue
                        
                        # Save overlaid image
                        img_filename = os.path.basename(full_img_path)
                        img_name, img_ext = os.path.splitext(img_filename)
                        overlay_filename = f"{img_name}_overlay{img_ext}"
                        overlay_path = os.path.join(overlay_images_dir, overlay_filename)
                        
                        cv2.imwrite(overlay_path, img)
                        images_overlaid += 1
                        
                    except Exception as e:
                        images_failed += 1
                        if images_failed <= 5:  # Only print first few errors
                            print(f"    Warning: Failed to overlay keypoints on {img_path}: {e}")
                else:
                    images_failed += 1
            
            print(f"✓ Overlaid keypoints on {images_overlaid} images")
            if images_failed > 0:
                print(f"  Failed to process {images_failed} images")
            print(f"  Overlaid images saved to: {overlay_images_dir}")
            
        except Exception as e:
            print(f"Warning: Could not overlay keypoints: {e}")
            import traceback
            traceback.print_exc()
        
        # Compute pixel error between predictions and ground truth
        print("\nComputing pixel errors...")
        try:
            def pixel_error(keypoints_true: np.ndarray, keypoints_pred: np.ndarray) -> np.ndarray:
                """Root mean square error between true and predicted keypoints.

                Args:
                    keypoints_true: shape (samples, n_keypoints, 2)
                    keypoints_pred: shape (samples, n_keypoints, 2)
                Returns:
                    shape (samples, n_keypoints)
                """
                error = np.linalg.norm(keypoints_true - keypoints_pred, axis=2)
                return error
            
            # Load ground truth
            df_gt = pd.read_csv(ood_csv_file, header=[0, 1, 2], index_col=0)
            
            # Get unique bodyparts from ground truth
            gt_bodyparts = df_gt.columns.get_level_values('bodyparts').unique()
            print(f"Ground truth bodyparts ({len(gt_bodyparts)}): {gt_bodyparts[:5]}...")
            
            # Build a mapping from bodypart to prediction columns
            # Predictions CSV now has structure: (scorer, bodyparts, coords)
            pred_cols_by_bodypart = {}
            
            for col in df_pred.columns:
                if isinstance(col, tuple) and len(col) == 3:
                    # (scorer, bodyparts, coords)
                    scorer, bodypart, coord = col
                    if bodypart not in pred_cols_by_bodypart:
                        pred_cols_by_bodypart[bodypart] = {}
                    pred_cols_by_bodypart[bodypart][coord] = col
            
            print(f"Prediction bodyparts found ({len(pred_cols_by_bodypart)}): {list(pred_cols_by_bodypart.keys())[:5]}...")
            
            # Find common bodyparts between GT and predictions
            # Match bodyparts (handling potential name variations)
            common_bodyparts = []
            bodypart_mapping = {}  # Maps GT bodypart to prediction bodypart
            
            for gt_bp in gt_bodyparts:
                # Try exact match first
                if gt_bp in pred_cols_by_bodypart:
                    common_bodyparts.append(gt_bp)
                    bodypart_mapping[gt_bp] = gt_bp
                else:
                    # Try with base name (remove suffixes)
                    base_gt_bp = gt_bp.split('.')[0] if '.' in str(gt_bp) else gt_bp
                    for pred_bp, coords_dict in pred_cols_by_bodypart.items():
                        pred_bp_base = pred_bp.split('.')[0] if isinstance(pred_bp, str) and '.' in str(pred_bp) else str(pred_bp)
                        if pred_bp_base == base_gt_bp and 'x' in coords_dict and 'y' in coords_dict:
                            common_bodyparts.append(gt_bp)
                            bodypart_mapping[gt_bp] = pred_bp
                            break
            
            if not common_bodyparts:
                print("Warning: No matching bodyparts found between GT and predictions")
                raise ValueError("No matching bodyparts")
            
            print(f"Common bodyparts ({len(common_bodyparts)}): {common_bodyparts[:5]}...")
            
            # Verify alignment: print mapping for first few bodyparts
            print(f"\nBodypart alignment verification (first 10):")
            for i, gt_bp in enumerate(common_bodyparts[:10]):
                pred_bp = bodypart_mapping[gt_bp]
                match_indicator = "✓" if gt_bp == pred_bp else f"→ {pred_bp}"
                print(f"  {i:2d}. GT: {gt_bp:15s} {match_indicator}")
            
            # Verify that all GT bodyparts are matched
            unmatched_gt = set(gt_bodyparts) - set(common_bodyparts)
            if unmatched_gt:
                print(f"\n⚠ Warning: {len(unmatched_gt)} GT bodyparts not found in predictions: {list(unmatched_gt)[:5]}")
            
            # Verify that prediction columns exist for all mapped bodyparts
            missing_cols = []
            for gt_bp in common_bodyparts:
                pred_bp = bodypart_mapping[gt_bp]
                if pred_bp not in pred_cols_by_bodypart:
                    missing_cols.append((gt_bp, pred_bp))
                elif 'x' not in pred_cols_by_bodypart[pred_bp] or 'y' not in pred_cols_by_bodypart[pred_bp]:
                    missing_cols.append((gt_bp, pred_bp))
            
            if missing_cols:
                print(f"\n⚠ Warning: {len(missing_cols)} bodyparts missing required columns:")
                for gt_bp, pred_bp in missing_cols[:5]:
                    print(f"  GT: {gt_bp} → Pred: {pred_bp}")
            
            # Get common image paths (intersection of GT and prediction indices)
            # Filter out header rows
            gt_index_clean = [idx for idx in df_gt.index if not (isinstance(idx, str) and idx.lower() in ['coords', 'bodyparts', 'scorer'])]
            pred_index_clean = [idx for idx in df_pred.index if not (isinstance(idx, str) and idx.lower() in ['coords', 'bodyparts', 'scorer'])]
            
            # Convert to sets for faster lookup and to find intersection
            gt_index_set = set(gt_index_clean)
            pred_index_set = set(pred_index_clean)
            
            # Find common images (must exist in both)
            common_images_set = gt_index_set & pred_index_set
            common_images = [img for img in pred_index_clean if img in common_images_set]
            
            # Validate sizes
            print(f"GT images: {len(gt_index_clean)}, Prediction images: {len(pred_index_clean)}")
            print(f"Common images: {len(common_images)}")
            
            if not common_images:
                print("Warning: No matching images found between GT and predictions")
                raise ValueError("No matching images")
            
            # Check for images in GT but not in predictions
            missing_in_pred = gt_index_set - pred_index_set
            if missing_in_pred:
                print(f"Warning: {len(missing_in_pred)} images in GT but not in predictions (showing first 3):")
                for img in list(missing_in_pred)[:3]:
                    print(f"  - {img}")
            
            # Check for images in predictions but not in GT
            missing_in_gt = pred_index_set - gt_index_set
            if missing_in_gt:
                print(f"Warning: {len(missing_in_gt)} images in predictions but not in GT (showing first 3):")
                for img in list(missing_in_gt)[:3]:
                    print(f"  - {img}")
            
            print(f"Processing {len(common_images)} images with {len(common_bodyparts)} bodyparts...")
            
            # Prepare arrays for vectorized computation
            # Shape: (samples, n_keypoints, 2)
            n_samples = len(common_images)
            n_keypoints = len(common_bodyparts)
            
            print(f"Array shapes: keypoints_true=({n_samples}, {n_keypoints}, 2), keypoints_pred=({n_samples}, {n_keypoints}, 2)")
            
            keypoints_true = np.full((n_samples, n_keypoints, 2), np.nan)
            keypoints_pred = np.full((n_samples, n_keypoints, 2), np.nan)
            
            gt_scorer = df_gt.columns.get_level_values('scorer')[0]
            
            # Track statistics
            valid_keypoints_count = 0
            missing_keypoints_count = 0
            
            # Fill arrays
            for img_idx, img_path in enumerate(common_images):
                # Verify image exists in both DataFrames
                if img_path not in df_gt.index:
                    print(f"Warning: Image {img_path} not in GT index (should not happen)")
                    continue
                if img_path not in df_pred.index:
                    print(f"Warning: Image {img_path} not in prediction index (should not happen)")
                    continue
                
                for bp_idx, gt_bp in enumerate(common_bodyparts):
                    pred_bp = bodypart_mapping[gt_bp]
                    
                    # Get GT coordinates
                    gt_x_col = (gt_scorer, gt_bp, 'x')
                    gt_y_col = (gt_scorer, gt_bp, 'y')
                    
                    # Get prediction coordinates
                    pred_x_col = pred_cols_by_bodypart[pred_bp]['x']
                    pred_y_col = pred_cols_by_bodypart[pred_bp]['y']
                    
                    try:
                        # Use .loc to access values - this will raise KeyError if column doesn't exist
                        gt_x = df_gt.loc[img_path, gt_x_col]
                        gt_y = df_gt.loc[img_path, gt_y_col]
                        pred_x = df_pred.loc[img_path, pred_x_col]
                        pred_y = df_pred.loc[img_path, pred_y_col]
                        
                        # Only set if all values are valid
                        if pd.notna(gt_x) and pd.notna(gt_y) and pd.notna(pred_x) and pd.notna(pred_y):
                            keypoints_true[img_idx, bp_idx, 0] = float(gt_x)
                            keypoints_true[img_idx, bp_idx, 1] = float(gt_y)
                            keypoints_pred[img_idx, bp_idx, 0] = float(pred_x)
                            keypoints_pred[img_idx, bp_idx, 1] = float(pred_y)
                            valid_keypoints_count += 1
                        else:
                            missing_keypoints_count += 1
                    except (KeyError, ValueError, TypeError) as e:
                        missing_keypoints_count += 1
                        # Only print first few errors to avoid spam
                        if missing_keypoints_count <= 5:
                            print(f"  Warning: Could not get keypoint for {img_path}, {gt_bp}: {e}")
            
            print(f"Valid keypoints: {valid_keypoints_count}, Missing/NaN keypoints: {missing_keypoints_count}")
            
            # Sample verification: show a few keypoint values to verify alignment
            if len(common_images) > 0 and len(common_bodyparts) > 0:
                print(f"\nSample alignment check (first image, first 5 bodyparts):")
                sample_img_idx = 0
                sample_img = common_images[sample_img_idx]
                for bp_idx in range(min(5, len(common_bodyparts))):
                    gt_bp = common_bodyparts[bp_idx]
                    gt_x = keypoints_true[sample_img_idx, bp_idx, 0]
                    gt_y = keypoints_true[sample_img_idx, bp_idx, 1]
                    pred_x = keypoints_pred[sample_img_idx, bp_idx, 0]
                    pred_y = keypoints_pred[sample_img_idx, bp_idx, 1]
                    
                    if not (np.isnan(gt_x) or np.isnan(gt_y) or np.isnan(pred_x) or np.isnan(pred_y)):
                        error = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
                        print(f"  {gt_bp:15s}: GT=({gt_x:7.1f}, {gt_y:7.1f}), Pred=({pred_x:7.1f}, {pred_y:7.1f}), Error={error:6.2f}px")
                    else:
                        print(f"  {gt_bp:15s}: Missing data (GT or Pred has NaN)")
            
            # Validate array shapes before computation
            assert keypoints_true.shape == (n_samples, n_keypoints, 2), \
                f"keypoints_true shape mismatch: expected ({n_samples}, {n_keypoints}, 2), got {keypoints_true.shape}"
            assert keypoints_pred.shape == (n_samples, n_keypoints, 2), \
                f"keypoints_pred shape mismatch: expected ({n_samples}, {n_keypoints}, 2), got {keypoints_pred.shape}"
            
            # Compute pixel errors using vectorized function
            # The function handles NaN values automatically (will produce NaN where inputs are NaN)
            print(f"Computing pixel errors...")
            print(f"  Keypoints true shape: {keypoints_true.shape}")
            print(f"  Keypoints pred shape: {keypoints_pred.shape}")
            errors = pixel_error(keypoints_true, keypoints_pred)
            
            # Validate output shape
            expected_error_shape = (n_samples, n_keypoints)
            assert errors.shape == expected_error_shape, \
                f"Error shape mismatch: expected {expected_error_shape}, got {errors.shape}"
            print(f"  Errors shape: {errors.shape}")
            
            # Create DataFrame with same structure as before
            # Index: image paths, Columns: bodypart names
            # Ensure common_images and common_bodyparts match the array dimensions
            assert len(common_images) == n_samples, \
                f"common_images length mismatch: expected {n_samples}, got {len(common_images)}"
            assert len(common_bodyparts) == n_keypoints, \
                f"common_bodyparts length mismatch: expected {n_keypoints}, got {len(common_bodyparts)}"
            
            df_pixel_error = pd.DataFrame(
                errors,
                index=common_images,
                columns=common_bodyparts
            )
            
            # Validate DataFrame shape
            expected_df_shape = (n_samples, n_keypoints)
            assert df_pixel_error.shape == expected_df_shape, \
                f"DataFrame shape mismatch: expected {expected_df_shape}, got {df_pixel_error.shape}"
            
            # Save pixel error CSV
            # The example format has index as first column but without column name
            pixel_error_file = os.path.join(shuffle_dir, 'predictions_pixel_error_new.csv')
            df_pixel_error.index.name = None  # Remove index name to match example format
            df_pixel_error.to_csv(pixel_error_file, index=True)  # index=True to keep image paths as first column
            
            print(f"✓ Pixel errors saved to: {pixel_error_file}")
            print(f"  Shape: {df_pixel_error.shape} (expected: {expected_df_shape})")
            print(f"  Non-null errors: {df_pixel_error.notna().sum().sum()} / {df_pixel_error.size}")
            print(f"  Mean error per bodypart:")
            for col in df_pixel_error.columns:
                mean_err = df_pixel_error[col].mean()
                if pd.notna(mean_err):
                    print(f"    {col}: {mean_err:.2f} pixels")
                
        except Exception as e:
            print(f"Warning: Could not compute pixel errors: {e}")
            import traceback
            traceback.print_exc()
                
    
        print(f"✓ Inference complete!")
                


if __name__ == '__main__':
    """
    Usage examples:
    (dlc) python run_dlc.py --dataset=mirror-mouse --gpu_id=0 --train_frames=75
    (dlc) python run_dlc.py --dataset=mirror-mouse --gpu_id=0 --train_frames=100
    """

    parser = argparse.ArgumentParser()

    # base params
    parser.add_argument('--dataset', type=str, default='mirror-mouse',
                       help='Dataset name (default: mirror-mouse)')
    parser.add_argument('--gpu_id', default=0, type=int,
                       help='GPU ID to use for training (default: 0)')
    parser.add_argument('--train_frames', type=int, default=100,
                       help='Number of training frames (default: 100)')
    parser.add_argument('--skip_train', action='store_true',
                       help='Skip training and only run inference on OOD data (default: False)')

    namespace, _ = parser.parse_known_args()
    run_main(namespace)

#!/usr/bin/env python
"""
Setup DLC project for ibl-paw dataset.

This script:
1. Creates a DLC project from lightning-pose format data
2. Sets up train/test splits for specified number of frames
3. Creates training datasets for DLC
"""

import deeplabcut
import numpy as np
import os
import pandas as pd
import shutil
import yaml
import argparse

def get_dataset_info(dataset_name):
    """Get dataset information from litpose config."""
    global DATASET_NAME
    DATASET_NAME = dataset_name
    global WORKSPACE_ROOT
    WORKSPACE_ROOT = os.getcwd()
    global DLC_WORKING_DIR
    DLC_WORKING_DIR = os.path.join(WORKSPACE_ROOT, 'dlc_projects')
    global TRAIN_FRAMES
    TRAIN_FRAMES = 100
    global RNG_SEEDS
    RNG_SEEDS = [0, 1, 2]
    global TRAIN_PROB
    TRAIN_PROB = 0.95
    global TRAIN_BATCH_SIZE
    TRAIN_BATCH_SIZE = 16
    global LEARNING_RATE
    LEARNING_RATE = 1e-3
    global TRAIN_EPOCHS
    TRAIN_EPOCHS = 300
    global SCORER
    global DATA_DIR
    DATA_DIR = os.path.join(WORKSPACE_ROOT, 'data', DATASET_NAME)
    global VIDEO_DIR
    global LABELED_CSV_FILE
    global LABELED_DATA_DIR
    if dataset_name == 'ibl-paw':
        TRAIN_BATCH_SIZE = 16
        LEARNING_RATE = 1e-3
        TRAIN_EPOCHS = 300
        SCORER = 'lightning_tracker'
        VIDEO_DIR = os.path.join(DATA_DIR, 'videos')
        LABELED_CSV_FILE = 'CollectedData.csv'
        LABELED_DATA_DIR = os.path.join(DATA_DIR, 'labeled-data')
    elif dataset_name == 'crim13':
        TRAIN_BATCH_SIZE = 16
        LEARNING_RATE = 1e-3
        TRAIN_EPOCHS = 300
        SCORER = 'mt'
        VIDEO_DIR = os.path.join(DATA_DIR, 'videos_InD')
        LABELED_CSV_FILE = 'labels.csv'
        LABELED_DATA_DIR = os.path.join(DATA_DIR, 'labeled-data_InD')
    elif dataset_name == 'mirror-mouse':
        TRAIN_BATCH_SIZE = 16
        LEARNING_RATE = 1e-3
        TRAIN_EPOCHS = 300
        SCORER = 'rick'
        VIDEO_DIR = os.path.join(DATA_DIR, 'videos_InD')
        LABELED_CSV_FILE = 'CollectedData.csv'
        LABELED_DATA_DIR = os.path.join(DATA_DIR, 'labeled-data_InD')
    elif dataset_name == 'mirror-fish':
        TRAIN_BATCH_SIZE = 16
        LEARNING_RATE = 1e-3
        TRAIN_EPOCHS = 300
        SCORER = 'rick'
        VIDEO_DIR = os.path.join(DATA_DIR, 'videos_InD')
        LABELED_CSV_FILE = 'CollectedData.csv'
        LABELED_DATA_DIR = os.path.join(DATA_DIR, 'labeled-data_InD')
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    global LABELED_FOLDER
    LABELED_FOLDER = LABELED_DATA_DIR.split('/')[-1]

def setup_dlc_project():
    """Create DLC project and set up basic structure."""
    print(f"Setting up DLC project for {DATASET_NAME}...")
    
    # Create working directory if it doesn't exist
    os.makedirs(DLC_WORKING_DIR, exist_ok=True)
    
    # Find a video file to initialize the project
    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    if not videos:
        raise FileNotFoundError(f"No videos found in {VIDEO_DIR}")
    
    # Use first video to initialize project
    init_video = os.path.join(VIDEO_DIR, videos[0])
    print(f"Using {videos[0]} to initialize project...")
    
    # Create DLC project
    dlc_config_path = deeplabcut.create_new_project(
        DATASET_NAME, 
        SCORER, 
        [init_video], 
        DLC_WORKING_DIR, 
        copy_videos=False, 
        multianimal=False
    )
    dlc_dir = os.path.dirname(dlc_config_path)
    print(f"DLC project created at: {dlc_dir}, config path: {dlc_config_path}")
    
    return dlc_config_path, dlc_dir


def update_bodyparts(dlc_config_path):
    """Update DLC config with bodypart names from litpose data."""
    print("Updating bodypart names...")
    
    # Load labels from litpose format
    labels_file = os.path.join(DATA_DIR, LABELED_CSV_FILE)
    ptl_labels = pd.read_csv(labels_file, index_col=0, header=[0, 1, 2])
    
    # Extract keypoint names (bodyparts)
    keypoint_names = [b[1] for b in ptl_labels.columns if b[2] == 'x']
    print(f"Found {len(keypoint_names)} keypoints: {keypoint_names}")
    
    # Update DLC config
    dlc_config = yaml.safe_load(open(dlc_config_path))
    dlc_config['bodyparts'] = keypoint_names
    yaml.dump(dlc_config, open(dlc_config_path, 'w'))
    
    return keypoint_names, ptl_labels


def copy_labeled_data(dlc_dir, ptl_labels, dlc_config_path):
    """Copy labeled data from litpose format to DLC format."""
    print("Copying labeled data...")
    
    # Remove the default labeled-data directory
    
    labeled_data_dir = os.path.join(dlc_dir, 'labeled-data')
    if os.path.exists(labeled_data_dir):
        shutil.rmtree(labeled_data_dir)
    
    # Copy labeled data from source directory
    # Note: ibl-paw already uses 'labeled-data' (not 'labeled-data_InD')
    source_labeled_dir = LABELED_DATA_DIR
    shutil.copytree(source_labeled_dir, labeled_data_dir)
    print(f"Copied labeled data from {source_labeled_dir}")
    
    # Create per-video annotation H5 files
    videos = np.unique([p.split('/')[1] for p in ptl_labels.index if LABELED_FOLDER in p])
    video_paths = {}
    
    for video in videos:
        df_tmp = ptl_labels[ptl_labels.index.str.contains(video)].copy()
        # replace labeled-data*/ with labeled-data/
        df_tmp.index = df_tmp.index.str.replace(LABELED_FOLDER, 'labeled-data/')
        # Rename scorer level to match SCORER (e.g., 'lightning_tracker' -> 'mic')
        # if DATASET_NAME == 'ibl-paw':
        #     df_tmp.columns = df_tmp.columns.str.replace('lightning_tracker', SCORER)
        df_tmp.to_hdf(
            os.path.join(dlc_dir, 'labeled-data', video, 'CollectedData_%s.h5' % SCORER),
            key='df_with_missing', mode='w')
        video_path = os.path.join(VIDEO_DIR, video + '.mp4')
        video_paths[video_path] = {'crop': '0, 1000, 0, 1000'}
    # update dlc config file
    dlc_config = yaml.safe_load(open(dlc_config_path))
    dlc_config['video_sets'] = video_paths
    yaml.dump(dlc_config, open(dlc_config_path, 'w'))

    # get all video dirs in labeled_data_dir
    all_video_dirs = [f for f in os.listdir(labeled_data_dir) if os.path.isdir(os.path.join(labeled_data_dir, f))]
    # remove the unused videos
    for video_dir in all_video_dirs:
        if video_dir not in videos:
            shutil.rmtree(os.path.join(labeled_data_dir, video_dir))
            print(f"Removed unused video {video_dir} from {labeled_data_dir}")
    return dlc_config_path


def create_train_test_splits(ptl_labels, train_frames, rng_seeds, dlc_config_path, train_prob=0.95):
    """Create train/test splits for specified number of frames."""
    print(f"Creating train/test splits for {train_frames} frames...")
    
    # Note: ptl_labels should already be filtered to only InD data at this point
    # (from copy_labeled_data which now filters before saving CSV)
    # But we'll double-check to be safe
    ind_mask = ptl_labels.index.str.contains(LABELED_FOLDER)
    if not ind_mask.all():
        raise ValueError(f"CSV contains non-InD rows. Please check the labeled data directory and the CSV file.")
    
    # select the indices of the labeled data
    all_indices = ptl_labels.index[ind_mask].tolist()
    all_indices = np.arange(len(ptl_labels))[ind_mask]
    # all_indices_tuples = []
    # for index in all_indices:
    #     tuple_index = tuple(index.split('/'))
    #     all_indices_tuples.append(tuple_index)
    # all_indices = all_indices_tuples
    shuffle_names = []
    idxs_train = []
    idxs_test = []
    
    for rng_seed in rng_seeds:
        # Set random seed for reproducibility
        np.random.seed(rng_seed)
        
        # Shuffle indices
        shuffled_indices = np.random.permutation(all_indices)
        
        # Select first train_frames for training
        if train_frames <= len(shuffled_indices):
            train_indices = shuffled_indices[:train_frames].tolist()
            n_val = int(len(shuffled_indices) * (1 - train_prob))
            n_val = min(n_val, train_frames)
            val_indices = shuffled_indices[train_frames:train_frames+n_val].tolist()
        else:
            raise ValueError(f"Train frames ({train_frames}) exceeds available data ({len(shuffled_indices)})")
        
        shuffle_name = int(str(train_frames) + str(rng_seed))
        shuffle_names.append(shuffle_name)
        idxs_train.append(train_indices)
        idxs_test.append(val_indices)
        
        print(f"  Shuffle {shuffle_name}: {len(train_indices)} train, {len(val_indices)} val")
        print(f"    Train indices range: [{min(train_indices) if train_indices else 'N/A'}, {max(train_indices) if train_indices else 'N/A'}]")
        print(f"    Val indices range: [{min(val_indices) if val_indices else 'N/A'}, {max(val_indices) if val_indices else 'N/A'}]")
    
    config = yaml.safe_load(open(dlc_config_path))
    config['TrainingFraction'] = [1-int(len(val_indices) / (len(val_indices) + len(train_indices)) * 100) / 100] * len(rng_seeds)
    yaml.dump(config, open(dlc_config_path, 'w'))
    return shuffle_names, idxs_train, idxs_test


def create_training_datasets(dlc_config_path, shuffle_names, idxs_train, idxs_test):
    """Create DLC training datasets with specified splits."""
    print("Creating DLC training datasets...")
    vals = deeplabcut.create_training_dataset(
        dlc_config_path,
        augmenter_type='imgaug',
        Shuffles=shuffle_names,
        trainIndices=idxs_train,
        testIndices=idxs_test,
    )


def main(args):
    """Main function to set up DLC project."""
    
    dataset_name = args.dataset
    get_dataset_info(dataset_name)
    print("=" * 60)
    print(f"DLC Project Setup for {DATASET_NAME}")
    print(f"Training frames: {args.train_frames}")
    print(f"Random seeds: {RNG_SEEDS}")
    print("=" * 60)
    
    # Step 1: Create DLC project
    dlc_config_path, dlc_dir = setup_dlc_project()
    
    # Step 2: Update bodyparts
    keypoint_names, ptl_labels = update_bodyparts(dlc_config_path)

    # Step 3: Copy labeled data and create DLC-formatted CSV
    dlc_config_path = copy_labeled_data(dlc_dir, ptl_labels, dlc_config_path)
    
    # Step 4: Create train/test splits (use DLC-formatted labels)
    shuffle_names, idxs_train, idxs_test = create_train_test_splits(
        ptl_labels, args.train_frames, RNG_SEEDS, dlc_config_path, train_prob=TRAIN_PROB
    )
    
    # Step 5: Create training datasets
    create_training_datasets(
        dlc_config_path, shuffle_names, idxs_train, idxs_test
    )
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print(f"Config path: {dlc_config_path}")
    print(f"DLC directory: {dlc_dir}")
    print("\nNext steps:")
    print(f"1. Review the config file: {dlc_config_path}")
    print(f"2. Run training with: python run_dlc.py --dataset={DATASET_NAME} --train_frames={args.train_frames} --gpu_id=0")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setup DLC project for litpose dataset')
    parser.add_argument('--train_frames', type=int, default=100,
                       help=f'Number of training frames (default: 100)')
    parser.add_argument('--dataset', type=str, default='mirror-mouse',
                       help=f'Dataset name (default: mirror-mouse)')
    args = parser.parse_args()
    np.random.seed(42)
    main(args)


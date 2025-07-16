from beast.extraction import (
    export_frames,
    select_frame_idxs_kmeans
)
import os
import json
from utils.utils import (
    get_args,
    get_video_paths_by_id,
    get_anchor_view_paths,
    get_all_views_for_anchor,
    get_video_id_from_path,
    set_seed
)
from pathlib import Path
import numpy as np
from datetime import datetime

def main():
    args = get_args()
    set_seed(args.seed)
    dataset = args.dataset
    input_dir = os.path.join(args.input_dir, dataset, 'videos')
    output_dir = os.path.join(args.output_dir, dataset)
    n_digits = 8
    extension = 'png'
    # First get all video paths grouped by ID
    video_dict = get_video_paths_by_id(input_dir)
    # get first key of video_dict
    # Get available views from the first video entry
    first_video_views = next(iter(video_dict.values()))
    avail_views = list(first_video_views.keys())
    print(f"Available views: {avail_views}")
    if dataset == 'mirror-mouse-separate':
        # 17 videos, 2 views per video
        # 2 * 17 * 3000 = 102000 frames
        frames_per_video = 3000
        anchor_view = 'top'
        # get video paths only for anchor view
        anchor_video_paths = get_anchor_view_paths(video_dict, anchor_view)
    else:
        raise ValueError(f'Dataset {dataset} not supported')

    print(f"Extracting frames from {len(anchor_video_paths)} videos from {anchor_view} view in {dataset} dataset")
    for video_path in anchor_video_paths:
        all_view_video_paths = get_all_views_for_anchor(video_path, video_dict)
        anchor_idxs = select_frame_idxs_kmeans(
            video_file=video_path,
            resize_dims=32,
            n_frames_to_select=frames_per_video,
        )
        video_id = get_video_id_from_path(video_path)
        # sort anchor_idxs
        anchor_idxs = np.sort(anchor_idxs)
        frames_to_label = np.array([
            "img%s.%s" % (str(idx).zfill(n_digits), extension) for idx in anchor_idxs
        ])

        for view, view_video_path in all_view_video_paths.items():
            output_view_dir = Path(output_dir) / video_id / view
            output_view_dir = output_view_dir.resolve()
            print(f"Exporting frames for {view} view to {output_view_dir}")
            # export frames
            # export_frames(
            #     video_file=view_video_path,
            #     frame_idxs=anchor_idxs,
            #     output_dir=output_view_dir,
            #     context_frames=0,
            # )
        # save a csv file with the frames_to_label
        csv_path = Path(output_dir) / video_id / "selected_frames.csv"
        np.savetxt(csv_path, frames_to_label, delimiter=",", fmt="%s")
        
    video_ids = list(video_dict.keys())
    # save information about available views and anchor view into output_dir
    # add timestamp to the info.json file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create a dictionary with all the dataset information
    dataset_info = {
        "dataset": dataset,
        "description": "A self-supervised dataset for multi-view learning",
        "available_views": avail_views,
        "anchor_view": anchor_view,
        "video_ids": video_ids,
        "number_of_videos": len(video_ids),
        "input_directory": input_dir,
        "output_directory": str(output_dir),
        "frames_per_video": frames_per_video,
        "n_digits": n_digits,
        "extension": extension,
        "timestamp": timestamp,
        "author": args.author,
        "seed": args.seed
    }
    
    # Save as JSON file
    with open(Path(output_dir) / "info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

if __name__ == "__main__":
    main()
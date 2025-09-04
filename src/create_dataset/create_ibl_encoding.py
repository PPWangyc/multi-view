import os
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
from one.api import ONE

from utils.ibl_utils import (
    prepare_data,
    bin_spiking_data,
    list_brain_regions,
    select_brain_regions,
    load_video_index,
    load_keypoints,
    load_video,
    resize_video,
    load_keypoints
)
from utils.utils import (
    set_seed,
    get_args,
)

# ---------------
# PREPROCESS DATA
# ---------------

def main(args):
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international", 
        silent=True,
        cache_dir='data/ibl'
    )

    params = {
        "interval_len": 2, 
        "binsize": 0.02, 
        "single_region": False,
        "align_time": 'stimOn_times', 
        "time_window": (-.5, 1.5), 
        "fr_thresh": 0.5
    }

    N_WORKERS = 4

    ENCODING_TEST_EIDS_PATH = "data/encoding_test_eids.txt"
    IDXS_DIR = Path("data/idxs")

    with open(ENCODING_TEST_EIDS_PATH, "r") as f:
        eids = f.read().splitlines()

    ANCHOR_CAMERA = "left"

    # Random select 100 integers between 0 to 119, make sure there are no duplicates
    RANDOM_VID_IDX = random.sample(range(120), 100)
    # sort the random indices
    RANDOM_VID_IDX.sort()

    OUTPUT_DIR = Path("data/encoding/ibl-mouse-separate")

    avail_views = ['left', 'right']

    print(f"Number of test encoding sessions: {len(eids)}")
    print(eids)

    eid = args.eid
    assert eid in eids, f"eid {eid} not in encoding test eids"
    eids = [eid]

    # Get encoding sessions
    for eid_idx, eid in enumerate(eids):
        neural_dict, _, meta_data, trials_data, _ = prepare_data(
                    one, eid, n_workers=N_WORKERS
        )
        
        regions, beryl_reg = list_brain_regions(neural_dict, **params)
        region_cluster_ids = select_brain_regions(neural_dict, beryl_reg, regions, **params)
        binned_spikes, _, intervals = bin_spiking_data(
            region_cluster_ids,
            neural_dict,
            trials_df=trials_data['trials_df'],
            n_workers=1,
            **params
        )
        # Keep responsive neurons
        mean_fr = binned_spikes.sum(1).mean(0) / params["interval_len"]
        keep_unit_idxs = np.argwhere(mean_fr > 1/params["fr_thresh"]).flatten()
        binned_spikes = binned_spikes[..., keep_unit_idxs]

        # load video index list
        anchor_video_index_list, _ = load_video_index(one, eid, ANCHOR_CAMERA, intervals)
        # load all view urls
        view_url_dict = {view: load_video_index(one, eid, view, only_url=True) for view in avail_views}

        # load all view indices
        view_idxs_dict = {}
        for view in avail_views:
            if view == ANCHOR_CAMERA:
                view_idxs_dict[view] = anchor_video_index_list
                continue
            # open the indices file which end with .npy
            idxs_path = IDXS_DIR / f"_ibl_{view}.indices.{eid}.npy"
            view_idxs = np.load(idxs_path)
            view_idxs_list = []
            for i in range(len(intervals)):
                _interval_idx = anchor_video_index_list[i]
                view_interval_idxs = view_idxs[_interval_idx].tolist()
                # select only the random video indices
                view_idxs_list.append(view_interval_idxs)
            view_idxs_dict[view] = view_idxs_list

        # process each trial
        for trial_idx in tqdm(range(len(intervals)), desc=f"Processing trial for session {eid} ({eid_idx+1}/{len(eids)})"):
            # trial spike
            spike = binned_spikes[trial_idx]
            video_views, kp_views = {}, {}
            for view in avail_views:
                trial_video = load_video(
                    index=view_idxs_dict[view][trial_idx],
                    url=view_url_dict[view]
                )
            
                # select only the random video indices
                trial_video = trial_video[RANDOM_VID_IDX]
                _, h, w = trial_video.shape
                h = w = 224
                trial_video = resize_video(
                    video=trial_video,
                    height=h,
                    width=w
                )
                # add view-video to dict
                video_views[view] = trial_video
                # deal with keypoints
                kps = load_keypoints(one, eid, view)
                trial_kps ={}
                for key, kp in kps.items():
                    trial_kps[key] = kp[view_idxs_dict[view][trial_idx]].to_numpy()
                    trial_kps[key] = trial_kps[key][RANDOM_VID_IDX]
                    # check of there is any nan value in the keypoint
                    assert not np.isnan(trial_kps[key]).any(), f'Nan value in keypoint {key}'
                kp_views[view] = trial_kps
            data = {
                'video': video_views,
                'keypoints': kp_views,
                'spike': spike,
                'meta': {
                    'eid': eid,
                    'trial_idx': trial_idx,
                    'interval': intervals[trial_idx],
                }
            }
            save_path = os.path.join(OUTPUT_DIR, eid, f"{trial_idx}.npy")
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, data)
if __name__ == "__main__":
    args = get_args()
    set_seed(42)
    main(args)
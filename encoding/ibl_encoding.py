from utils.utils import (
    get_args,
    set_seed,
    NAME_MODEL
)
import safetensors
from data.datasets import EncodingDataset
from utils.log_utils import get_logger

import os
from accelerate import Accelerator
from beast.io import load_config
import torch
from tqdm import tqdm
import numpy as np

def main(args):
    config = load_config(args.config)
    eid = args.eid
    logger.info(f"Running encoding with experiment ID: {eid}")

    data_dir = os.path.join(args.data_dir, eid)

    # accelerate
    accelerator = Accelerator()
    modes = ['train', 'val', 'test']
    datasets = [EncodingDataset(data_dir=data_dir, imgaug_pipeline=None, mode=mode) for mode in modes]
    avail_views = list(datasets[0][0]['input_video_view'].keys())
    config['data']['avail_views'] = avail_views
    config['model']['model_params']['num_views'] = len(avail_views)
    logger.info(f"Available views: {avail_views}")

    # model
    model = NAME_MODEL[config['model']['name']](config)
    # Handle resuming from checkpoint
    if config['training']['resume'] is not None:
        model_ckpt = safetensors.torch.load_file(config['training']['resume'])
        model.load_state_dict(model_ckpt, strict=False)
        logger.info(f"Resumed model from checkpoint: {config['training']['resume']}")
    else:
        logger.warning("Training model from scratch.")

    model = accelerator.prepare(model)
    meta = {
        'eid': eid,
        'model': config['model']['name'],
        'resume': config['training']['resume'],
    }
    # load encoding_dict according to metadata if exists
    # ...
    encoding_dict = {mode: {view: [] for view in avail_views} for mode in modes}
    # inference to get embeddings
    with torch.no_grad():
        model.eval()
        for (mode, dataset) in zip(modes, datasets):
            logger.info(f"Running inference on {mode} set...")
            for trial_data in tqdm(dataset):
                for view in trial_data['input_video_view'].keys():
                    input_dict = {'image': trial_data['input_video_view'][view].to(accelerator.device)} # 
                    results_dict = model.predict_step(input_dict)
                    embeddings = results_dict['latents'].cpu().numpy()
                    encoding_dict[mode][view].append(embeddings)
            # Concatenate embeddings for each view
            for view in avail_views:
                encoding_dict[mode][view] = np.concatenate(encoding_dict[mode][view], axis=0)
            logger.info(f"Completed inference on {mode} set.")
    # Save embeddings
    save_dir = os.path.join(data_dir, 'embeddings', config['model']['name'])
    os.makedirs(save_dir, exist_ok=True)
    # train encoding models with best validation hyperparameters BPS
    # ...
    

    # test encoding models
    # ...

if __name__ == "__main__":
    args = get_args()
    set_seed(42) # For reproducibility
    logger = get_logger()
    main(args)
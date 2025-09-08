from utils.utils import (
    get_args,
    set_seed,
    NAME_MODEL,
    create_encoding_log,
    train_rrr_with_tune,
    train_tcn_with_tune
    
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
    T = datasets[0][0]['spike'].shape[0]  # Assuming all trials have the same Time steps
    config['data']['avail_views'] = avail_views
    config['model']['model_params']['num_views'] = len(avail_views)
    logger.info(f"Available views: {avail_views}")

    metadata = {
        'ds': 'ibl-mouse-separate',
        'eid': eid,
        'model': config['model']['name'],
        'resume': config['training']['resume'],
    }
    # create a log dir based on metadata
    log_dir = create_encoding_log(metadata)
    # load encoding_dict according to metadata if exists
    if os.path.exists(os.path.join(log_dir, 'encoding_dict.npy')):
        encoding_dict = np.load(os.path.join(log_dir, 'encoding_dict.npy'), allow_pickle=True).item()
        logger.warning(f"Loaded existing encoding_dict from {os.path.join(log_dir, 'encoding_dict.npy')}")
    else:
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
        logger.info(f"Creating new encoding_dict at {os.path.join(log_dir, 'encoding_dict.npy')}")
        encoding_dict = {mode: {view: [] for view in avail_views} for mode in modes}
        # inference to get embeddings
        with torch.no_grad():
            model.eval()
            for (mode, dataset) in zip(modes, datasets):
                logger.info(f"Running inference on {mode} set...")
                spike = []
                for trial_data in tqdm(dataset):
                    for view in trial_data['input_video_view'].keys():
                        input_dict = {'image': trial_data['input_video_view'][view].to(accelerator.device)} # 
                        results_dict = model.predict_step(input_dict)
                        embeddings = results_dict['latents'].cpu().numpy()
                        encoding_dict[mode][view].append(embeddings)
                    spike.append(trial_data['spike'].numpy())
                encoding_dict[mode]['spike'] = np.concatenate(spike, axis=0).reshape(len(dataset), T, -1)  
                # Concatenate embeddings for each view
                for view in avail_views:
                    encoding_dict[mode][view] = np.concatenate(encoding_dict[mode][view], axis=0).reshape(len(dataset), T, -1)
                logger.info(f"Completed inference on {mode} set.")
        encoding_dict['eid'] = eid
        encoding_dict['avail_views'] = avail_views
        # Save embeddings
        np.save(os.path.join(log_dir, 'encoding_dict.npy'), encoding_dict)
    # train encoding models with best validation hyperparameters BPS
    # train/val/test rrr with tuning
    rrr_result = train_rrr_with_tune(encoding_dict, 2)
    logger.info(f"RRR Encoding {eid} Test BPS: {rrr_result[eid]['bps']:.4f}, R2: {rrr_result[eid]['r2']:.4f}, VE: {rrr_result[eid]['ve']:.4f}")
    # train/val/test tcn with tuning
    tcn_result = train_tcn_with_tune(encoding_dict, 2)
    logger.info(f"TCN Encoding {eid} Test BPS: {tcn_result[eid]['bps']:.4f}, R2: {tcn_result[eid]['r2']:.4f}, VE: {tcn_result[eid]['ve']:.4f}")
    results_dict = {
        'rrr': rrr_result,
        'cnn': None,
        'log_dir': log_dir,
        'eid': eid,
        'avail_views': avail_views,
    }
    np.save(os.path.join(log_dir, 'results_dict.npy'), results_dict)

    

if __name__ == "__main__":
    args = get_args()
    set_seed(42) # For reproducibility
    logger = get_logger()
    main(args)
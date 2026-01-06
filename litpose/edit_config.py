import yaml
from utils.utils import (
    get_args,
)
import os
import sys
import torch
import pandas as pd

def main(args):
    model_type = args.model_type
    # get available gpus
    num_gpus = torch.cuda.device_count()
    
    # Validate required arguments
    if not hasattr(args, 'litpose_config') or not args.litpose_config:
        print("Error: --litpose_config is required", file=sys.stderr)
        sys.exit(1)
    
    litpose_config_path = args.litpose_config
    if not os.path.exists(litpose_config_path):
        print(f"Error: Config file not found: {litpose_config_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        with open(litpose_config_path, 'r') as f:
            litpose_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error: Failed to load config file: {e}", file=sys.stderr)
        sys.exit(1)
    
    model = args.model
    train_frame = args.litpose_frame
    dataset_name = args.dataset
    epochs = args.epochs
    mode = args.mode
    seed = args.seed
    data_dir = args.data_dir if hasattr(args, 'data_dir') and args.data_dir else None
    
    # Validate data_dir if provided
    if data_dir and not os.path.exists(data_dir):
        print(f"Warning: Data directory does not exist: {data_dir}", file=sys.stderr)
    
    # Set unfreezing epoch based on mode
    if mode == 'ft':
        # finetune mode: train the whole model from unfreeze_epoch to max_epochs
        unfreezing_epoch = 20
    elif mode == 'lp':
        # linear probe mode: freeze the backbone and only train the head
        unfreezing_epoch = epochs
    else:
        print(f"Warning: Unknown mode '{mode}', defaulting to finetune mode", file=sys.stderr)
        unfreezing_epoch = 20
    
    # Determine save directory - use output_base_dir if provided, otherwise use default
    base_dir = os.getcwd()
    if hasattr(args, 'output_base_dir') and args.output_base_dir:
        output_base_dir = args.output_base_dir
    else:
        # Default to outputs directory in current working directory
        output_base_dir = os.path.join(base_dir, 'outputs')

    # load learning rate from hyper_table.csv
    hyper_table_path = os.path.join(base_dir, 'configs', 'litpose', 'litpose_hyper.csv')
    hyper_table = pd.read_csv(hyper_table_path)
    learning_rate = float(hyper_table.loc[(hyper_table['model'] == model) & (hyper_table['dataset'] == dataset_name) & (hyper_table['model_type'] == model_type), 'learning_rate'].values[0])
    epochs = int(hyper_table.loc[(hyper_table['model'] == model) & (hyper_table['dataset'] == dataset_name) & (hyper_table['model_type'] == model_type), 'epochs'].values[0])
    file_name = f"ds-{dataset_name}_mode-{mode}_model-{model}_type-{model_type}_frame-{train_frame}_epoch-{epochs}_seed-{seed}"
    save_dir = os.path.join(output_base_dir, file_name)
    # Handle special models (-mv, -sv, -mvt, -mvt)
    if any(model.endswith(suffix) for suffix in ['-mv', '-sv', '-mvt', '-mvt', '-beast', '-beast-c', '-mae']):
        model_arch = model.split('-')[0]
        model_path = os.path.join(base_dir, f"logs/ds-{dataset_name}_model-{model}")
        if model.endswith('-beast'):
            model_path = os.path.join(model_path, 'vitb-beast.pth')
        elif model.endswith('-beast-c'):
            model_path = os.path.join(model_path, 'vitb-beast-c.pth')
        else:
            model_path = os.path.join(model_path, 'model.safetensors')
        if not os.path.exists(model_path):
            print(f"Warning: Pretrained model not found: {model_path}", file=sys.stderr)
        if model_arch == 'vits':
            model_version = 'dino'
        elif model_arch == 'vitb':
            model_version = 'imagenet'
        else:
            raise ValueError(f"Unknown model architecture: {model_arch}")
        if 'dinov3' in model:
            model_version = 'dinov3'
        elif 'dinov2' in model:
            model_version = 'dinov2'
        model = f'{model_arch}_{model_version}'
        # Set the pretrain model path
        if 'model' not in litpose_config:
            litpose_config['model'] = {}
        litpose_config['model']['backbone_checkpoint'] = model_path
        print(f"Pretrained model path: {model_path}", file=sys.stderr)
    elif model == 'vggt':
        model = 'vitl_dinov2'
        print(f"VGGT model detected, using {model} as the model", file=sys.stderr)
    if model_type == 'heatmap':
        final_ratio = 0.0
    elif model_type == 'heatmap_multiview_transformer':
        final_ratio = 0.5
    elif model_type == 'heatmap_multiview_aggregator':
        final_ratio = 0.5
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    # Add patch mask
    if 'patch_mask' not in litpose_config['training']:
        litpose_config['training']['patch_mask'] = {}
    litpose_config['training']['patch_mask']['init_epoch'] = 40
    litpose_config['training']['patch_mask']['final_epoch'] = epochs
    litpose_config['training']['patch_mask']['init_ratio'] = 0.0
    litpose_config['training']['patch_mask']['final_ratio'] = final_ratio
    
    # Edit config - update data directory
    if data_dir:
        if 'data' not in litpose_config:
            litpose_config['data'] = {}
        litpose_config['data']['data_dir'] = data_dir
        # litpose_config['data']['video_dir'] = os.path.join(data_dir, 'videos')
        # litpose_config['data']['camera_params_file'] = os.path.join(data_dir, 'calibrations.csv')
    
    # Update seed
    litpose_config['training']['rng_seed_data_pt'] = seed
    litpose_config['training']['rng_seed_model_pt'] = seed
    
    if 'dali' not in litpose_config:
        litpose_config['dali'] = {}
    if 'general' not in litpose_config['dali']:
        litpose_config['dali']['general'] = {}
    litpose_config['dali']['general']['seed'] = seed
    
    # Edit mode
    litpose_config['training']['unfreezing_epoch'] = unfreezing_epoch
    
    # Change model
    if 'model' not in litpose_config:
        litpose_config['model'] = {}
    litpose_config['model']['backbone'] = model
    litpose_config['model']['model_type'] = model_type
    
    # Change train frame
    litpose_config['training']['train_frames'] = train_frame
    
    # Change max epoch
    litpose_config['training']['max_epochs'] = epochs
    
    # Edit optimizer - ensure optimizer_params exists
    if 'optimizer_params' not in litpose_config['training']:
        litpose_config['training']['optimizer_params'] = {}
    litpose_config['training']['optimizer_params']['learning_rate'] = learning_rate
    
    # edit num gpus
    litpose_config['training']['num_gpus'] = num_gpus
    
    # Change save dir
    if 'hydra' not in litpose_config:
        litpose_config['hydra'] = {}
    if 'run' not in litpose_config['hydra']:
        litpose_config['hydra']['run'] = {}
    litpose_config['hydra']['run']['dir'] = save_dir
    
    # Save edited config
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    config_name = f'{file_name}.yaml'
    output_path = os.path.join(output_dir, config_name)
    
    try:
        with open(output_path, 'w') as f:
            yaml.dump(litpose_config, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f"Error: Failed to save config file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Output the output_path to make the script capture the output
    print(output_path)

if __name__ == '__main__':
    args = get_args()
    main(args)

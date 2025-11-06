import yaml
from utils.utils import (
    get_args,
)
import os
import sys
import torch

def main(args):
    model_type = args.model_type
    # get available gpus
    available_gpus = torch.cuda.device_count()
    # make sure available gpus num is divisible by 2, if gpus > 1
    if available_gpus > 1 and available_gpus % 2 != 0:
        print(f"Warning: Available GPUs number is not divisible by 2, using {available_gpus - 1} GPUs", file=sys.stderr)
        available_gpus = available_gpus - 1

    num_gpus = max(available_gpus, 4)
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
    
    file_name = f"ds-{dataset_name}_mode-{mode}_model-{model}_type-{model_type}_frame-{train_frame}_epoch-{epochs}_seed-{seed}"
    
    # Determine save directory - use output_base_dir if provided, otherwise use default
    base_dir = os.getcwd()
    if hasattr(args, 'output_base_dir') and args.output_base_dir:
        output_base_dir = args.output_base_dir
    else:
        # Default to outputs directory in current working directory
        output_base_dir = os.path.join(base_dir, 'outputs')
    
    save_dir = os.path.join(output_base_dir, file_name)
    
    # Add patch mask
    if 'patch_mask' not in litpose_config:
        litpose_config['patch_mask'] = {}
    # if model == 'vits_dino' and model_type == 'heatmap_multiview_transformer':
    #     final_ratio = 0.5
    # else:
    #     final_ratio = 0.0
    # litpose_config['patch_mask']['init_epoch'] = 40
    # litpose_config['patch_mask']['final_epoch'] = epochs
    # litpose_config['patch_mask']['init_ratio'] = 0.0
    # litpose_config['patch_mask']['final_ratio'] = final_ratio

    # Handle special models (-mv, -sv, -mvt, -mvt)
    if any(model.endswith(suffix) for suffix in ['-mv', '-sv', '-mvt', '-mvt', '-beast']):
        model_arch = model.split('-')[0]
        model_path = os.path.join(base_dir, f"logs/ds-{dataset_name}_model-{model}")
        if model.endswith('-beast'):
            model_path = os.path.join(model_path, 'vitb-beast.pth')
        else:
            model_path = os.path.join(model_path, 'model.safetensors')
        if not os.path.exists(model_path):
            print(f"Warning: Pretrained model not found: {model_path}", file=sys.stderr)
        model = f'{model_arch}_imagenet'
        # Set the pretrain model path
        if 'model' not in litpose_config:
            litpose_config['model'] = {}
        litpose_config['model']['backbone_checkpoint'] = model_path
        print(f"Pretrained model path: {model_path}", file=sys.stderr)

    # Edit config - update data directory
    if data_dir:
        if 'data' not in litpose_config:
            litpose_config['data'] = {}
        litpose_config['data']['data_dir'] = data_dir
        # litpose_config['data']['video_dir'] = os.path.join(data_dir, 'videos')
        # litpose_config['data']['camera_params_file'] = os.path.join(data_dir, 'calibrations.csv')
    
    # Update seed
    if 'training' not in litpose_config:
        litpose_config['training'] = {}
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

    # Set learning rate based on model type
    if 'vit' in model:
        learning_rate = 5e-5
        if dataset_name == 'fly-anipose':
            if 'vitb' in model:
                learning_rate = 2e-4
    else:
        learning_rate = 1e-3
    
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

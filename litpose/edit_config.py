import yaml
from utils.utils import (
    get_args,
)
import os

def main(args):
    litpose_config_path = args.litpose_config
    with open(litpose_config_path, 'r') as f:
        litpose_config = yaml.safe_load(f)
    model = args.model
    train_frame = args.litpose_frame
    dataset_name = args.dataset
    epochs = args.epochs
    mode= args.mode
    seed = args.seed
    if 'vit' in model:
        # double the epochs for vit models
        # epochs = epochs * 2
        # batch_size = litpose_config['training']['train_batch_size']
        # litpose_config['training']['train_batch_size'] = batch_size * 2
        learning_rate = 5e-5
    else:
        learning_rate = 1e-3
    if mode == 'ft':
        # finetune mode: train the whole model from unfreeze_epoch to max_epochs
        unfreezing_epoch = 20
    elif mode == 'lp':
        # linear probe mode
        # freeze the backbone and only train the head
        unfreezing_epoch = epochs
    file_name = f"ds-{dataset_name}_mode-{mode}_model-{model}_frame-{train_frame}_epoch-{epochs}_seed-{seed}"
    save_dir = os.path.join(
        "/scratch/yl6624/video-spike/outputs",
        file_name
    )
    base_dir = os.getcwd()
    if model in ['vitb-mv', 'vitb-sv']:
        model_path = os.path.join(base_dir, f"logs/ds-{dataset_name}_model-{model}/model.safetensors")
        model = 'vitb_imagenet'
        # set the pretrain model path
        litpose_config['model']['backbone_checkpoint'] = model_path
    # edit config
    # change seed
    litpose_config['training']['rng_seed_data_pt'] = seed
    litpose_config['training']['rng_seed_model_pt'] = seed
    litpose_config['dali']['general']['seed'] = seed
    # edit mode
    litpose_config['training']['unfreezing_epoch'] = unfreezing_epoch
    # change model
    litpose_config['model']['backbone'] = model
    # if model in ['vit_cm', 'vit_m']:
    #     # change image augmentation
    #     litpose_config['training']['imgaug'] = 'default'
    # change train frame
    litpose_config['training']['train_frames'] = train_frame
    # change max epoch
    litpose_config['training']['max_epochs'] = epochs
    # edit optimizer
    # edit learning rate
    litpose_config['training']['optimizer_params']['learning_rate'] = learning_rate
    # change save dir
    litpose_config['hydra']['run']['dir'] = save_dir
    # save edited config
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    config_name = f'{file_name}.yaml'
    output_path = os.path.join(output_dir, config_name)
    with open(output_path, 'w') as f:
        yaml.dump(litpose_config, f)
    # output the output_path to make my script capture the output
    print(output_path)
if __name__ == '__main__':
    args = get_args()
    main(args)
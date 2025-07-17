from beast.io import load_config
from beast.data.augmentations import imgaug_pipeline, expand_imgaug_str_to_dict
from utils.utils import (
    set_seed,
    get_args,
    NAME_MODEL,
    plot_example_images,
)
from data.datasets import MVDataset
from utils.log_utils import get_logger
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
# accelerate
from accelerate import Accelerator

logger = get_logger()

def main():
    args = get_args()
    set_seed(args.seed)
    config = load_config(args.config)

    # accelerate
    accelerator = Accelerator()

    # imgaug transform
    pipe_params = config.get('training', {}).get('imgaug', 'none')
    if isinstance(pipe_params, str):
        pipe_params = expand_imgaug_str_to_dict(pipe_params)
    imgaug_pipeline_ = imgaug_pipeline(pipe_params)
    # dataset
    dataset = MVDataset(
        data_dir=config.get('data', {}).get('data_dir'),
        imgaug_pipeline=None,
    )
    config['model']['model_params']['num_views'] = len(dataset.available_views)
    config['data']['avail_views'] = dataset.available_views
    
    # dataloader
    dataloader = DataLoader(dataset, batch_size=config.get('training', {}).get('train_batch_size'), shuffle=True)
    # model
    model = NAME_MODEL[config['model']['name']](config)
    # optimizer
    epochs = config.get('training').get('num_epochs')
    lr = config['optimizer']['lr']
    weight_decay = config['optimizer']['wd']
    total_steps = epochs * len(dataloader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,
        total_steps=total_steps,
        pct_start=config['optimizer']['warmup_pct'],
        final_div_factor=1,
    )
    model, dataloader, optimizer, scheduler = accelerator.prepare(model, dataloader, optimizer, scheduler)
    # train
    for epoch in range(epochs):
        model.train()
        running_loss = 0.
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
        for batch in pbar:
            optimizer.zero_grad()
            results_dict = model(batch)
            loss = results_dict['loss']
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # Plot example images every 100 steps
            if pbar.n % 100 == 0 and accelerator.is_main_process:
                plot_example_images(batch, results_dict, recon_num=8, save_path=f'plots/epoch_{epoch+1}_step_{pbar.n}.png')
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        print(f'Epoch {epoch+1} loss: {running_loss/len(dataloader)}')
        

if __name__ == '__main__':
    main()
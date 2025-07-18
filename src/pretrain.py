from beast.io import load_config
from beast.data.augmentations import imgaug_pipeline, expand_imgaug_str_to_dict
from utils.utils import (
    set_seed,
    get_args,
    NAME_MODEL,
    plot_example_images,
    create_log_dir,
    load_checkpoint_for_resume,
    get_resume_checkpoint_path,
    save_all_training_info,
)
from data.datasets import MVDataset
from utils.log_utils import get_logger
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
# accelerate
from accelerate import Accelerator
import os
import json

logger = get_logger()

def main():
    args = get_args()
    set_seed(args.seed)
    config = load_config(args.config)

    # Create log directory
    experiment_name = config.get('experiment_name', 'mae_pretrain')
    log_dir = create_log_dir(experiment_name)
    logger.info(f"Log directory created: {log_dir}")

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
    train_batch_size = config['training']['train_batch_size']
    
    # dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=train_batch_size, 
        shuffle=True, 
        num_workers=config['training']['num_workers'], 
        pin_memory=True,
        drop_last=True,
    )
    # model
    model = NAME_MODEL[config['model']['name']](config)
    # optimizer
    epochs = config.get('training').get('num_epochs')
    lr = config['optimizer']['lr']
    # get world size
    world_size = accelerator.num_processes
    global_batch_size = train_batch_size * world_size
    weight_decay = config['optimizer']['wd']
    
    log_every_n_epochs = config['training']['log_every_n_epochs'] # log every n epochs
    expected_batch_size = config['training']['expected_batch_size']
    if expected_batch_size is not None:
        accumulate_grad_batches = max(1, expected_batch_size // global_batch_size)
    else:
        accumulate_grad_batches = max(1, config['optimizer']['accumulate_grad_batches'])
    total_steps = epochs * len(dataloader) // world_size // accumulate_grad_batches
    lr = lr * global_batch_size / 256 * accumulate_grad_batches # scale lr by global batch size

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
    
    # Model saving variables
    best_loss = float('inf')
    best_epoch = 0
    start_epoch = 0
    
    # Handle resuming from checkpoint
    if config['training']['resume'] is not None:
        checkpoint_path = get_resume_checkpoint_path(config['training']['resume'], config['training']['resume_from_best'])
        training_state = load_checkpoint_for_resume(
            checkpoint_path, accelerator, model, optimizer, scheduler, logger
        )
        
        # Update training state
        start_epoch = training_state.get('epoch', 0)
        best_loss = training_state.get('best_loss', float('inf'))
        best_epoch = training_state.get('best_epoch', 0)
        
        logger.info(f"Resuming from epoch {start_epoch}")
        logger.info(f"Best loss so far: {best_loss:.4f} at epoch {best_epoch}")
        
        # If resuming, use the same log directory as the checkpoint
        if os.path.isdir(config['training']['resume']):
            log_dir = config['training']['resume']
            logger.info(f"Using existing log directory: {log_dir}")
    # Collect training configuration information
    training_info = {
        "epochs": epochs,
        "total_steps": total_steps,
        "learning_rate": lr,
        "global_batch_size": global_batch_size,
        "local_batch_size": train_batch_size,
        "world_size": world_size,
        "accumulate_grad_batches": accumulate_grad_batches,
        "effective_batch_size": global_batch_size * accumulate_grad_batches,
        "dataset_size": len(dataset),
        "steps_per_epoch": len(dataloader),
        "available_views": dataset.available_views,
        "num_views": len(dataset.available_views),
        "weight_decay": weight_decay,
        "warmup_percentage": config['optimizer']['warmup_pct'],
        "scheduler_type": "OneCycleLR",
        "optimizer_type": "AdamW",
        "model_name": config['model']['name'],
        "seed": args.seed,
        "experiment_name": experiment_name
    }
    
    # Save all training information to log directory
    save_all_training_info(config, training_info, args, log_dir, logger)
    
    # Log training configuration
    logger.info("=" * 50)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 50)
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Learning rate: {lr:.2e}")
    logger.info(f"Global batch size: {global_batch_size} (local: {train_batch_size} × {world_size} processes)")
    logger.info(f"Gradient accumulation steps: {accumulate_grad_batches}")
    logger.info(f"Effective batch size: {global_batch_size * accumulate_grad_batches}")
    logger.info(f"Dataset size: {len(dataset)} samples")
    logger.info(f"Steps per epoch: {len(dataloader)}")
    logger.info(f"Available views: {dataset.available_views}")
    logger.info(f"Model: {config['model']['name']}")
    logger.info("=" * 50)
    # train
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
        for batch_idx, batch in enumerate(pbar):
            # Forward pass
            results_dict = model(batch)
            loss = results_dict['loss']
            running_loss += loss.item()
            
            # Scale loss for gradient accumulation
            loss = loss / accumulate_grad_batches
            loss.backward()
            
            # Update weights every accumulate_grad_batches steps
            if (batch_idx + 1) % accumulate_grad_batches == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{loss.item() * accumulate_grad_batches:.4f}'})
        
        avg_loss = running_loss/len(dataloader)
        logger.info(f'Epoch {epoch+1} loss: {avg_loss}')
        
        if epoch % log_every_n_epochs == 0 or epoch == epochs - 1 or epoch == 0:
            # save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch + 1
                if accelerator.is_main_process:
                    best_model_path = os.path.join(log_dir, "checkpoints", "best_model.pth")
                    accelerator.save_state(best_model_path)
                    logger.info(f"Best model saved at epoch {epoch+1} with loss: {best_loss:.4f}")
            
            # Save last model
            if accelerator.is_main_process:
                last_model_path = os.path.join(log_dir, "checkpoints", "last_model.pth")
                accelerator.save_state(last_model_path)
                
                # Save training state
                training_state = {
                    'epoch': epoch + 1,
                    'best_loss': best_loss,
                    'best_epoch': best_epoch,
                    'global_step': (epoch + 1) * len(dataloader)
                }
                training_state_path = os.path.join(log_dir, "checkpoints", "training_state.json")
                with open(training_state_path, 'w') as f:
                    json.dump(training_state, f, indent=2)
                
                # Save plots
                plot_path = os.path.join(log_dir, "plots", f'epoch_{epoch+1}_step_{pbar.n}.png')
                plot_example_images(batch, results_dict, recon_num=8, save_path=plot_path)
    
    # Final summary
    if accelerator.is_main_process:
        logger.info(f"Training completed. Best model saved at epoch {best_epoch} with loss: {best_loss:.4f}")
        logger.info(f"All checkpoints and plots saved in: {log_dir}")

if __name__ == '__main__':
    main()
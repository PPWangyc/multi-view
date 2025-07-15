from transformers import (
    IJepaConfig, 
    IJepaModel,
    AutoModel,
    ViTMAEModel,
    ViTMAEConfig,
    ViTModel,
    ViTConfig,
    AutoImageProcessor,
)
import torch
import torch.nn as nn
from utils.log_utils import get_logger
from PIL import Image

logger = get_logger()

def transfer_vit_mae_to_ijepa(ijepa_config, vit_mae_model_name="facebook/vit-mae-base"):
    """
    Transfer weights from a ViT-MAE model to an IJEPA model.
    
    Args:
        ijepa_config (IJepaConfig): Configuration for IJEPA model
        vit_mae_model_name (str): Name of the pretrained ViT-MAE model
    
    Returns:
        IJepaModel: IJEPA model with transferred weights
    """
    logger.info(f"Loading ViT-MAE model: {vit_mae_model_name}")
    
    # Load ViT-MAE model and config
    vit_mae_model = ViTMAEModel.from_pretrained(vit_mae_model_name)
    vit_mae_config = ViTMAEConfig.from_pretrained(vit_mae_model_name)
    
    logger.info(f"ViT-MAE config: hidden_size={vit_mae_config.hidden_size}, "
          f"num_hidden_layers={vit_mae_config.num_hidden_layers}, "
          f"num_attention_heads={vit_mae_config.num_attention_heads}")
    
    logger.info(f"IJEPA config: hidden_size={ijepa_config.hidden_size}, "
          f"num_hidden_layers={ijepa_config.num_hidden_layers}, "
          f"num_attention_heads={ijepa_config.num_attention_heads}")
    
    # Create IJEPA model
    ijepa_model = IJepaModel(ijepa_config)
    
    # Transfer weights
    logger.info("Transferring weights from ViT-MAE to IJEPA...")
    
    # Get state dicts
    vit_mae_state_dict = vit_mae_model.state_dict()
    ijepa_state_dict = ijepa_model.state_dict()
    
    # Create mapping of parameter names
    # The architectures should be compatible, so we can map most parameters directly
    transferred_count = 0
    skipped_count = 0
    
    for ijepa_name, ijepa_param in ijepa_state_dict.items():
        if ijepa_name in vit_mae_state_dict:
            # Check if shapes match
            if vit_mae_state_dict[ijepa_name].shape == ijepa_param.shape:
                ijepa_state_dict[ijepa_name] = vit_mae_state_dict[ijepa_name].clone()
                transferred_count += 1
                logger.info(f"Transferred: {ijepa_name}")
            else:
                # Handle position embedding mismatch: MAE has CLS token, IJEPA doesn't
                if (ijepa_name == "embeddings.position_embeddings" and 
                    vit_mae_state_dict[ijepa_name].shape[1] == ijepa_param.shape[1] + 1 and
                    vit_mae_state_dict[ijepa_name].shape[0] == ijepa_param.shape[0] and
                    vit_mae_state_dict[ijepa_name].shape[2] == ijepa_param.shape[2]):
                    # Skip the first CLS token position embedding from MAE
                    ijepa_state_dict[ijepa_name] = vit_mae_state_dict[ijepa_name][:, 1:, :].clone()
                    transferred_count += 1
                    logger.info(f"Transferred (skipped CLS): {ijepa_name}")
                else:
                    logger.warning(f"Shape mismatch for {ijepa_name}: "
                          f"ViT-MAE {vit_mae_state_dict[ijepa_name].shape} vs "
                          f"IJEPA {ijepa_param.shape}")
                    skipped_count += 1
        else:
            logger.warning(f"Parameter not found in ViT-MAE: {ijepa_name}")
            skipped_count += 1
    
    # Load the transferred state dict
    ijepa_model.load_state_dict(ijepa_state_dict, strict=False)
    
    logger.info(f"Weight transfer completed!")
    logger.info(f"Transferred: {transferred_count} parameters")
    logger.info(f"Skipped: {skipped_count} parameters")
    
    return ijepa_model

def transfer_dino_to_ijepa(ijepa_config, dino_model_name="facebook/dino-vitb16"):
    """
    Transfer weights from a DINO ViT-B-16 model to an IJEPA model.
    
    Args:
        ijepa_config (IJepaConfig): Configuration for IJEPA model
        dino_model_name (str): Name of the pretrained DINO model
    
    Returns:
        IJepaModel: IJEPA model with transferred weights
    """
    logger.info(f"Loading DINO model: {dino_model_name}")
    
    # Load DINO model and config
    dino_model = ViTModel.from_pretrained(dino_model_name)
    dino_config = ViTConfig.from_pretrained(dino_model_name)
    
    logger.info(f"DINO config: hidden_size={dino_config.hidden_size}, "
          f"num_hidden_layers={dino_config.num_hidden_layers}, "
          f"num_attention_heads={dino_config.num_attention_heads}")
    
    logger.info(f"IJEPA config: hidden_size={ijepa_config.hidden_size}, "
          f"num_hidden_layers={ijepa_config.num_hidden_layers}, "
          f"num_attention_heads={ijepa_config.num_attention_heads}")
    
    # Create IJEPA model
    ijepa_model = IJepaModel(ijepa_config)
    
    # Transfer weights
    logger.info("Transferring weights from DINO to IJEPA...")
    
    # Get state dicts
    dino_state_dict = dino_model.state_dict()
    ijepa_state_dict = ijepa_model.state_dict()
    
    # Create mapping of parameter names
    # DINO and IJEPA should have compatible architectures
    transferred_count = 0
    skipped_count = 0
    
    for ijepa_name, ijepa_param in ijepa_state_dict.items():
        if ijepa_name in dino_state_dict:
            # Check if shapes match
            if dino_state_dict[ijepa_name].shape == ijepa_param.shape:
                ijepa_state_dict[ijepa_name] = dino_state_dict[ijepa_name].clone()
                transferred_count += 1
                logger.info(f"Transferred: {ijepa_name}")
            else:
                # Handle position embedding mismatch: DINO has CLS token, IJEPA doesn't
                if (ijepa_name == "embeddings.position_embeddings" and 
                    dino_state_dict[ijepa_name].shape[1] == ijepa_param.shape[1] + 1 and
                    dino_state_dict[ijepa_name].shape[0] == ijepa_param.shape[0] and
                    dino_state_dict[ijepa_name].shape[2] == ijepa_param.shape[2]):
                    # Skip the first CLS token position embedding from DINO
                    ijepa_state_dict[ijepa_name] = dino_state_dict[ijepa_name][:, 1:, :].clone()
                    transferred_count += 1
                    logger.info(f"Transferred (skipped CLS): {ijepa_name}")
                else:
                    logger.warning(f"Shape mismatch for {ijepa_name}: "
                          f"DINO {dino_state_dict[ijepa_name].shape} vs "
                          f"IJEPA {ijepa_param.shape}")
                    skipped_count += 1
        else:
            logger.warning(f"Parameter not found in DINO: {ijepa_name}")
            skipped_count += 1
    
    # Load the transferred state dict
    ijepa_model.load_state_dict(ijepa_state_dict, strict=False)
    
    logger.info(f"Weight transfer completed!")
    logger.info(f"Transferred: {transferred_count} parameters")
    logger.info(f"Skipped: {skipped_count} parameters")
    
    return ijepa_model

def main():
    ijepa_config = IJepaConfig()
    
    # Example usage for ViT-MAE transfer
    logger.info("=== ViT-MAE to IJEPA Transfer ===")
    model_ijepa = transfer_vit_mae_to_ijepa(ijepa_config)
    logger.info(f"Final model type: {type(model_ijepa)}")
    logger.info(f"Model config: {model_ijepa.config.model_type}")
    logger.info("\n" + "="*50 + "\n")
    
    # Example usage for DINO transfer
    # logger.info("=== DINO to IJEPA Transfer ===")
    # model_ijepa = transfer_dino_to_ijepa(ijepa_config)
    # logger.info(f"Final model type: {type(model_ijepa)}")
    # logger.info(f"Model config: {model_ijepa.config.model_type}")
    # logger.info("\n" + "="*50 + "\n")

    from datasets import load_dataset
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
    inputs = image_processor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model_ijepa(**inputs)
    print(outputs['last_hidden_state'].shape)

if __name__ == "__main__":
    main()
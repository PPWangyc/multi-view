from transformers import (
    IJepaConfig, 
    IJepaModel,
    AutoModel,
    ViTMAEModel,
    ViTMAEConfig,
)
import torch
import torch.nn as nn

def transfer_vit_mae_to_ijepa(ijepa_config, vit_mae_model_name="facebook/vit-mae-base"):
    """
    Transfer weights from a ViT-MAE model to an IJEPA model.
    
    Args:
        vit_mae_model_name (str): Name of the pretrained ViT-MAE model
        ijepa_config (IJepaConfig, optional): Configuration for IJEPA model. 
                                             If None, will try to match ViT-MAE config.
    
    Returns:
        IJepaModel: IJEPA model with transferred weights
    """
    print(f"Loading ViT-MAE model: {vit_mae_model_name}")
    
    # Load ViT-MAE model and config
    vit_mae_model = ViTMAEModel.from_pretrained(vit_mae_model_name)
    vit_mae_config = ViTMAEConfig.from_pretrained(vit_mae_model_name)
    
    print(f"ViT-MAE config: hidden_size={vit_mae_config.hidden_size}, "
          f"num_hidden_layers={vit_mae_config.num_hidden_layers}, "
          f"num_attention_heads={vit_mae_config.num_attention_heads}")
        
    print(f"IJEPA config: hidden_size={ijepa_config.hidden_size}, "
          f"num_hidden_layers={ijepa_config.num_hidden_layers}, "
          f"num_attention_heads={ijepa_config.num_attention_heads}")
    
    # Create IJEPA model
    ijepa_model = IJepaModel(ijepa_config)
    
    # Transfer weights
    print("Transferring weights from ViT-MAE to IJEPA...")
    
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
                print(f"Transferred: {ijepa_name}")
            else:
                print(f"Shape mismatch for {ijepa_name}: "
                      f"ViT-MAE {vit_mae_state_dict[ijepa_name].shape} vs "
                      f"IJEPA {ijepa_param.shape}")
                skipped_count += 1
        else:
            print(f"Parameter not found in ViT-MAE: {ijepa_name}")
            skipped_count += 1
    
    # Load the transferred state dict
    ijepa_model.load_state_dict(ijepa_state_dict, strict=False)
    
    print(f"Weight transfer completed!")
    print(f"Transferred: {transferred_count} parameters")
    print(f"Skipped: {skipped_count} parameters")
    
    return ijepa_model

def main():
    ijepa_config = IJepaConfig()
    # Example usage
    model = transfer_vit_mae_to_ijepa(ijepa_config)
    print(f"Final model type: {type(model)}")
    print(f"Model config: {model.config.model_type}")

if __name__ == "__main__":
    main()
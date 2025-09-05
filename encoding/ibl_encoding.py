from utils.utils import (
    get_args,
    set_seed
)
from data.datasets import EncodingDataset
from utils.log_utils import get_logger

import os
from accelerate import Accelerator
from torch.utils.data import DataLoader

def main(args):
    eid = args.eid
    logger.info(f"Running encoding with experiment ID: {eid}")

    data_dir = os.path.join(args.data_dir, eid)

    train_dataset = EncodingDataset(data_dir=data_dir, imgaug_pipeline=None, mode='train')
    val_dataset = EncodingDataset(data_dir=data_dir, imgaug_pipeline=None, mode='val')
    test_dataset = EncodingDataset(data_dir=data_dir, imgaug_pipeline=None, mode='test')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    for batch in train_dataloader:
        for k, v in batch.items():
            if isinstance(v, dict):
                for vk, vv in v.items():
                    if isinstance(vv, dict):
                        for vvk, vvv in vv.items():
                            print(f"train {k}.{vk}.{vvk}: {vvv.shape}")
                    else:
                        print(f"train {k}.{vk}: {vv.shape}")
            else:
                print(f"train {k}: {v.shape}")
        break

    # accelerate
    accelerator = Accelerator()

    # train encoding models with best validation hyperparameters BPS
    # ...
    

    # test encoding models
    # ...

if __name__ == "__main__":
    args = get_args()
    set_seed(42) # For reproducibility
    logger = get_logger()
    main(args)
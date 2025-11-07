import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from typeguard import typechecked

from data.data_types import EncodingDict, ExampleDict, MultiViewDict
from utils.log_utils import get_logger

logger = get_logger()

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225] 

@typechecked
class BaseDataset(torch.utils.data.Dataset):
    """Base dataset that contains images."""

    def __init__(self, data_dir: str | Path, imgaug_pipeline: Callable | None) -> None:
        """Initialize a dataset for autoencoder models.

        Parameters
        ----------
        data_dir: absolute path to data directory
        imgaug_transform: imgaug transform pipeline to apply to images

        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise ValueError(f'{self.data_dir} is not a directory')

        self.imgaug_pipeline = imgaug_pipeline
        # collect ALL png files in data_dir
        self.image_list = sorted(list(self.data_dir.rglob('*.png')))
        if len(self.image_list) == 0:
            raise ValueError(f'{self.data_dir} does not contain image data in png format')

        # send image to tensor, resize to canonical dimensions, and normalize
        pytorch_transform_list = [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
        self.pytorch_transform = transforms.Compose(pytorch_transform_list)

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int | list) -> ExampleDict | list[ExampleDict]:
        """Get item(s) from dataset.

        Parameters
        ----------
        idx: single index or list  indices

        Returns
        -------
        Single ExampleDict or list of ExampleDict objects

        """
        # Handle batch of indices
        if isinstance(idx, list):
            return [self._get_single_item(i) for i in idx]
        else:
            # Handle single index
            return self._get_single_item(idx)

    def _get_single_item(self, idx: int) -> ExampleDict:
        """Get a single item from the dataset."""
        img_path = self.image_list[idx]

        # read image from file and apply transformations (if any)
        # if 1 color channel, change to 3.
        image = Image.open(img_path).convert('RGB')
        if self.imgaug_pipeline is not None:
            # expands add batch dim for imgaug
            transformed_images = self.imgaug_pipeline(images=np.expand_dims(image, axis=0))
            # get rid of the batch dim
            transformed_images = transformed_images[0]
        else:
            transformed_images = image

        transformed_images = self.pytorch_transform(transformed_images)

        return ExampleDict(
            image=transformed_images,  # shape (3, img_height, img_width)
            video=img_path.parts[-2],
            idx=idx,
            image_path=str(img_path),
        )

@typechecked
class MVDataset(torch.utils.data.Dataset):
    """Multi-view dataset that contains images."""

    def __init__(self, data_dir: str | Path, imgaug_pipeline: Callable | None) -> None:
        """Initialize a multi-view dataset.

        Parameters
        ----------
        data_dir: absolute path to data directory
        imgaug_transform: imgaug transform pipeline to apply to images

        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise ValueError(f'{self.data_dir} is not a directory')

        # read info.txt
        with open(self.data_dir / 'info.json', 'r') as f:
            self.info = json.load(f)
        self.available_views = self.info['available_views']
        self.video_ids = self.info['video_ids']

        # read all csv files in data_dir
        self.csv_files = sorted(list(self.data_dir.rglob('*.csv')))
        if len(self.csv_files) == 0:
            raise ValueError(f'{self.data_dir} does not contain csv files')

        # unique frame ids: video_id/frame_id
        self.unique_frame_ids = []
        for csv_file in self.csv_files:
            video_id = csv_file.parts[-2]
            df = pd.read_csv(csv_file)
            # get video_id from csv_file
            # df only has one column, and every row is a frame id
            frame_ids = df.iloc[:, 0].tolist()
            # add video_id to frame_ids
            frame_ids = [f'{video_id}/{frame_id}' for frame_id in frame_ids]
            # add to frame_ids
            self.unique_frame_ids.extend(frame_ids)
        # unique frame_ids
        # assert len(self.unique_frame_ids) == len(set(self.unique_frame_ids)), 'frame_ids are not unique'
        self.unique_frame_ids = list(set(self.unique_frame_ids))
        total_frames = len(self.unique_frame_ids) * len(self.available_views)
        logger.info(f'Dataset Summary:')
        logger.info(f'  • Unique frame IDs: {len(self.unique_frame_ids)}')
        logger.info(f'  • Available views: {self.available_views} ({len(self.available_views)} total)')
        logger.info(f'  • Anchor view: {self.info["anchor_view"]}')
        logger.info(f'  • Videos: {len(self.video_ids)}')
        logger.info(f'  • Total frames across all views: {total_frames}')

        self.imgaug_pipeline = imgaug_pipeline

        # make a dictionary of frame_ids and their paths
        self.frame_id_to_path = {}
        for unique_frame_id in self.unique_frame_ids:
            video_id = unique_frame_id.split('/')[0]
            frame_id = unique_frame_id.split('/')[1]
            # get the path to the frame
            self.frame_id_to_path[unique_frame_id] = {}
            for view in self.available_views:
                frame_path = self.data_dir / video_id / view / f'{frame_id}'
                self.frame_id_to_path[unique_frame_id][view] = frame_path
        
        if len(self.unique_frame_ids) == 0:
            raise ValueError(f'{self.data_dir} does not contain any multi-view data')

        # send image to tensor, resize to canonical dimensions, and normalize
        pytorch_transform_list = [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
        self.pytorch_transform = transforms.Compose(pytorch_transform_list)

    def __len__(self) -> int:
        return len(self.unique_frame_ids)

    def __getitem__(self, idx: int | list) -> MultiViewDict | list[MultiViewDict]:
        """Get item(s) from dataset.

        Parameters
        ----------
        idx: single index or list  indices

        Returns
        -------
        Single MultiViewDict or list of MultiViewDict objects

        """
        # Handle batch of indices
        if isinstance(idx, list):
            return [self._get_single_item(i) for i in idx]
        else:
            # Handle single index
            return self._get_single_item(idx)

    def _get_single_item(self, idx: int) -> MultiViewDict:
        """Get a single item from the dataset."""
        unique_frame_id = self.unique_frame_ids[idx]

        # random a number between 0 and len(self.available_views)
        input_view_idx = np.random.randint(0, len(self.available_views))
        input_view = self.available_views[input_view_idx]
        input_view_path = self.frame_id_to_path[unique_frame_id][input_view]
        
        output_view_idx = np.random.randint(0, len(self.available_views))
        output_view = self.available_views[output_view_idx]
        output_view_path = self.frame_id_to_path[unique_frame_id][output_view]

        # read image from file and apply transformations (if any)
        # if 1 color channel, change to 3.
        input_image = Image.open(input_view_path).convert('RGB')
        output_image = Image.open(output_view_path).convert('RGB')
        if self.imgaug_pipeline is not None:
            # expands add batch dim for imgaug
            input_transformed_images = self.imgaug_pipeline(images=np.expand_dims(input_image, axis=0))
            output_transformed_images = self.imgaug_pipeline(images=np.expand_dims(output_image, axis=0))
            # get rid of the batch dim
            input_transformed_images = input_transformed_images[0]
            output_transformed_images = output_transformed_images[0]
        else:
            input_transformed_images = input_image
            output_transformed_images = output_image

        input_transformed_images = self.pytorch_transform(input_transformed_images)
        output_transformed_images = self.pytorch_transform(output_transformed_images)

        return MultiViewDict(
            input_image=input_transformed_images,  # shape (3, img_height, img_width)
            output_image=output_transformed_images,  # shape (3, img_height, img_width)
            video_id=unique_frame_id.split('/')[0],
            frame_id=unique_frame_id.split('/')[1],
            idx=idx,
            input_image_path=str(input_view_path),
            output_image_path=str(output_view_path),
            input_view=input_view,
            output_view=output_view,
        )

class MVTDataset(torch.utils.data.Dataset):
    """Multi-view dataset that contains images."""

    def __init__(self, data_dir: str | Path, imgaug_pipeline: Callable | None) -> None:
        """Initialize a multi-view dataset.

        Parameters
        ----------
        data_dir: absolute path to data directory
        imgaug_transform: imgaug transform pipeline to apply to images

        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise ValueError(f'{self.data_dir} is not a directory')

        # read info.txt
        with open(self.data_dir / 'info.json', 'r') as f:
            self.info = json.load(f)
        self.available_views = sorted(self.info['available_views'])
        self.video_ids = self.info['video_ids']

        # read all csv files in data_dir
        self.csv_files = sorted(list(self.data_dir.rglob('*.csv')))
        if len(self.csv_files) == 0:
            raise ValueError(f'{self.data_dir} does not contain csv files')

        # unique frame ids: video_id/frame_id
        self.unique_frame_ids = []
        for csv_file in self.csv_files:
            video_id = csv_file.parts[-2]
            df = pd.read_csv(csv_file)
            # get video_id from csv_file
            # df only has one column, and every row is a frame id
            frame_ids = df.iloc[:, 0].tolist()
            # add video_id to frame_ids
            frame_ids = [f'{video_id}/{frame_id}' for frame_id in frame_ids]
            # add to frame_ids
            self.unique_frame_ids.extend(frame_ids)
        # unique frame_ids
        # assert len(self.unique_frame_ids) == len(set(self.unique_frame_ids)), 'frame_ids are not unique'
        self.unique_frame_ids = list(set(self.unique_frame_ids))
        total_frames = len(self.unique_frame_ids) * len(self.available_views)
        logger.info(f'Dataset Summary:')
        logger.info(f'  • Unique frame IDs: {len(self.unique_frame_ids)}')
        logger.info(f'  • Available views: {self.available_views} ({len(self.available_views)} total)')
        logger.info(f'  • Anchor view: {self.info["anchor_view"]}')
        logger.info(f'  • Videos: {len(self.video_ids)}')
        logger.info(f'  • Total frames across all views: {total_frames}')

        self.imgaug_pipeline = imgaug_pipeline

        # make a dictionary of frame_ids and their paths
        self.frame_id_to_path = {}
        for unique_frame_id in self.unique_frame_ids:
            video_id = unique_frame_id.split('/')[0]
            frame_id = unique_frame_id.split('/')[1]
            # get the path to the frame
            self.frame_id_to_path[unique_frame_id] = {}
            for view in self.available_views:
                frame_path = self.data_dir / video_id / view / f'{frame_id}'
                self.frame_id_to_path[unique_frame_id][view] = frame_path
        
        if len(self.unique_frame_ids) == 0:
            raise ValueError(f'{self.data_dir} does not contain any multi-view data')

        # send image to tensor, resize to canonical dimensions, and normalize
        pytorch_transform_list = [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
        self.pytorch_transform = transforms.Compose(pytorch_transform_list)

    def __len__(self) -> int:
        return len(self.unique_frame_ids)

    def __getitem__(self, idx: int | list) -> MultiViewDict | list[MultiViewDict]:
        """Get item(s) from dataset.

        Parameters
        ----------
        idx: single index or list  indices

        Returns
        -------
        Single MultiViewDict or list of MultiViewDict objects

        """
        # Handle batch of indices
        if isinstance(idx, list):
            return [self._get_single_item(i) for i in idx]
        else:
            # Handle single index
            return self._get_single_item(idx)

    def _get_single_item(self, idx: int) -> MultiViewDict:
        """Get a single item from the dataset."""
        unique_frame_id = self.unique_frame_ids[idx]

        # random a number between 0 and len(self.available_views)
        input_view_dict = {}
        for view in self.available_views:
            input_view_path = self.frame_id_to_path[unique_frame_id][view]
            input_image = Image.open(input_view_path).convert('RGB')
            if self.imgaug_pipeline is not None:
                # expands add batch dim for imgaug
                input_transformed_images = self.imgaug_pipeline(images=np.expand_dims(input_image, axis=0))
                # get rid of the batch dim
                input_transformed_images = input_transformed_images[0]
            else:
                input_transformed_images = input_image
            input_transformed_images = self.pytorch_transform(input_transformed_images)
            input_view_dict[view] = input_transformed_images
        input_image = torch.stack([input_view_dict[view] for view in self.available_views], dim=0) # shape (view, batch, channels, img_height, img_width)
        output_image = input_image.clone() # shape (view, batch, channels, img_height, img_width)
        return MultiViewDict(
            input_image=input_image,  # shape (view, batch, channels, img_height, img_width)
            output_image=output_image,  # shape (batch, view, channels, img_height, img_width)
            video_id=unique_frame_id.split('/')[0],
            frame_id=unique_frame_id.split('/')[1],
            idx=idx,
        )

@typechecked
class EncodingDataset(torch.utils.data.Dataset):
    """Multi-view dataset that contains images."""

    def __init__(self, data_dir: str | Path, imgaug_pipeline: Callable | None, mode: str) -> None:
        """Initialize a multi-view dataset.

        Parameters
        ----------
        data_dir: absolute path to data directory
        imgaug_transform: imgaug transform pipeline to apply to images

        """
        self.data_dir = Path(os.path.join(data_dir, mode))
        if not self.data_dir.is_dir():
            raise ValueError(f'{self.data_dir} is not a directory')

        # get npy files in data_dir
        self.npy_files = sorted(list(self.data_dir.rglob('*.npy')))
        if len(self.npy_files) == 0:
            raise ValueError(f'{self.data_dir} does not contain npy files')
        
        total_trials = len(self.npy_files)
        logger.info(f'Dataset Summary:')
        logger.info(f'  • Mode: {mode}')
        logger.info(f'  • Total trials (npy files): {total_trials}')

        self.imgaug_pipeline = imgaug_pipeline
        # send image to tensor, resize to canonical dimensions, and normalize
        pytorch_transform_list = [
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
        self.pytorch_transform = transforms.Compose(pytorch_transform_list)

    def __len__(self) -> int:
        return len(self.npy_files)

    def __getitem__(self, idx: int | list) -> EncodingDict | list[EncodingDict]:
        """Get item(s) from dataset.

        Parameters
        ----------
        idx: single index or list  indices

        Returns
        -------
        Single MultiViewDict or list of MultiViewDict objects

        """
        # Handle batch of indices
        if isinstance(idx, list):
            return [self._get_single_item(i) for i in idx]
        else:
            # Handle single index
            return self._get_single_item(idx)

    def _get_single_item(self, idx: int) -> EncodingDict:
        """Get a single item from the dataset."""
        data = np.load(self.npy_files[idx], allow_pickle=True).item()
        # process video views
        for view in data['video']:
            # read image from file and apply transformations (if any)
            # if 1 color channel, change to 3.
            data['video'][view] = torch.from_numpy(data['video'][view]).float()  
            data['video'][view] /= 255.0  # normalize to [0, 1]
            if len(data['video'][view].shape) == 3:
                # (T, H, W) -> (T, 1, H, W) -> (T, 3, H, W)
                data['video'][view] = data['video'][view].unsqueeze(1)
                data['video'][view] = data['video'][view].repeat(1, 3, 1, 1)
            # apply pytorch transforms
            if self.pytorch_transform:
                data['video'][view] = self.pytorch_transform(data['video'][view])
        # sort data['video'] by view name
        data['video'] = dict(sorted(data['video'].items()))
        data['keypoints'] = dict(sorted(data['keypoints'].items()))
        
        # process keypoints views
        input_keypoints_view = {}
        discrete_keypoints_dict = {}
        for view in data['keypoints']:
            # sort data['keypoints'][view] by keypoint name
            data['keypoints'][view] = dict(sorted(data['keypoints'][view].items()))
            view_kps = []
            for kp in data['keypoints'][view]:
                # kp is (T)
                data['keypoints'][view][kp] = torch.from_numpy(data['keypoints'][view][kp]).float()
                view_kps.append(data['keypoints'][view][kp])
            view_kps = torch.stack(view_kps, dim=1)  # (T, num_keypoints)
            input_keypoints_view[view] = view_kps
            discrete_keypoints_dict[view] = data['keypoints'][view]

        # process spike
        data['spike'] = torch.from_numpy(data['spike']).float()

        return EncodingDict(
            input_video_view=data['video'],
            input_keypoints_view=input_keypoints_view,
            input_discrete_keypoints_view=discrete_keypoints_dict,
            spike=data['spike'],
        )

def main():
    dataset = BaseDataset(data_dir='../../data/ssl/mirror-mouse-separate', imgaug_pipeline=None)
    print(dataset[0].keys())
    dataset = MVDataset(data_dir='../../data/ssl/mirror-mouse-separate', imgaug_pipeline=None)
    print(dataset[0].keys())
    dataset = EncodingDataset(data_dir='../../data/encoding/ibl-mouse-separate', imgaug_pipeline=None, mode='train')
    print(dataset[0].keys())

if __name__ == '__main__':
    main()
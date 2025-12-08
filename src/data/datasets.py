import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2
from typeguard import typechecked

from data.data_types import EncodingDict, ExampleDict, MultiViewDict
from utils.log_utils import get_logger

logger = get_logger()

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225] 

@typechecked
class BaseDataset(torch.utils.data.Dataset):
    """Base dataset that contains images."""

    def __init__(self, data_dir: str | Path, imgaug_pipeline: Callable | None, config: dict = None) -> None:
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
            v2.ToImage(),
            transforms.RandomResizedCrop(config['model']['model_params']['image_size'], scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
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
class VideoDataset(torch.utils.data.Dataset):
    """Video dataset that contains video clips around anchor frames."""

    def __init__(self, data_dir: str | Path, imgaug_pipeline: Callable | None, config: dict = None) -> None:
        """Initialize a video dataset.

        Parameters
        ----------
        data_dir: absolute path to data directory
        imgaug_pipeline: imgaug transform pipeline to apply to images

        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise ValueError(f'{self.data_dir} is not a directory')

        self.imgaug_pipeline = imgaug_pipeline
        # read all csv files in data_dir
        self.csv_files = sorted(list(self.data_dir.rglob('selected_frames.csv')))
        if len(self.csv_files) == 0:
            raise ValueError(f'{self.data_dir} does not contain selected_frames.csv files')

        # collect anchor frames from all csv files
        all_anchor_frames = []
        for csv_file in self.csv_files:
            video_id = csv_file.parts[-2]
            df = pd.read_csv(csv_file, header=None)
            # df only has one column, and every row is a frame id
            frame_ids = df.iloc[:, 0].tolist()
            for frame_id in frame_ids:
                all_anchor_frames.append((video_id, frame_id))

        if len(all_anchor_frames) == 0:
            raise ValueError(f'{self.data_dir} does not contain any anchor frames')

        # Filter anchor frames to only keep those where all 16 frames in the clip exist
        self.anchor_frames = []
        for video_id, anchor_frame_id in all_anchor_frames:
            if self._check_all_frames_exist(video_id, anchor_frame_id):
                self.anchor_frames.append((video_id, anchor_frame_id))

        if len(self.anchor_frames) == 0:
            raise ValueError(f'{self.data_dir} does not contain any valid anchor frames with all required frames')

        total_clips = len(self.anchor_frames)
        filtered_out = len(all_anchor_frames) - total_clips
        logger.info(f'VideoDataset Summary:')
        logger.info(f'  • Total anchor frames found: {len(all_anchor_frames)}')
        logger.info(f'  • Valid anchor frames (all frames exist): {total_clips}')
        logger.info(f'  • Filtered out (missing frames): {filtered_out}')
        logger.info(f'  • Clips per anchor: 16 frames (from ±8 frames around anchor)')

        # send image to tensor, resize to canonical dimensions, and normalize
        pytorch_transform_list = [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
        self.pytorch_transform = transforms.Compose(pytorch_transform_list)

    def _check_all_frames_exist(self, video_id: str, anchor_frame_id: str) -> bool:
        """Check if all 16 frames in the clip exist for a given anchor frame.
        
        Parameters
        ----------
        video_id: str
            Video identifier
        anchor_frame_id: str
            Anchor frame ID (e.g., "img00005469.png")
        
        Returns
        -------
        bool
            True if all 16 frames exist, False otherwise
        """
        # Extract frame number from anchor frame ID
        frame_number_str = anchor_frame_id.replace('img', '').replace('.png', '')
        anchor_frame_num = int(frame_number_str)

        # Get ±8 frames around anchor (17 frames total, then take first 16)
        for offset in range(-8, 8):  # -8 to +7 inclusive (16 frames)
            frame_num = anchor_frame_num + offset
            # Format frame number as 8-digit zero-padded string
            frame_id = f'img{frame_num:08d}.png'
            frame_path = self.data_dir / video_id / frame_id
            
            if not frame_path.exists():
                return False
        
        return True

    def __len__(self) -> int:
        return len(self.anchor_frames)

    def __getitem__(self, idx: int | list) -> ExampleDict | list[ExampleDict]:
        """Get item(s) from dataset.

        Parameters
        ----------
        idx: single index or list of indices

        Returns
        -------
        Single ExampleDict or list of ExampleDict objects
        The image field contains a tensor of shape (16, 3, 224, 224)

        """
        # Handle batch of indices
        if isinstance(idx, list):
            return [self._get_single_item(i) for i in idx]
        else:
            # Handle single index
            return self._get_single_item(idx)

    def _get_single_item(self, idx: int) -> ExampleDict:
        """Get a single item from the dataset."""
        video_id, anchor_frame_id = self.anchor_frames[idx]

        # Extract frame number from anchor frame ID (e.g., "img00005469.png" -> 5469)
        # Frame ID format: "img{frame_number:08d}.png"
        frame_number_str = anchor_frame_id.replace('img', '').replace('.png', '')
        anchor_frame_num = int(frame_number_str)

        # Get ±8 frames around anchor (17 frames total, then take first 16)
        clip_frame_nums = []
        for offset in range(-8, 9):  # -8 to +8 inclusive (17 frames)
            frame_num = anchor_frame_num + offset
            # Format frame number as 8-digit zero-padded string
            frame_id = f'img{frame_num:08d}.png'
            clip_frame_nums.append((frame_num, frame_id))

        # Load images for the clip (first 16 frames)
        # All frames should exist since we filtered during initialization
        clip_images = []
        clip_image_paths = []
        for frame_num, frame_id in clip_frame_nums[:16]:  # Take first 16 frames
            frame_path = self.data_dir / video_id / frame_id

            # All frames should exist, but check anyway for safety
            if not frame_path.exists():
                raise FileNotFoundError(f'Frame not found: {frame_path}. This should not happen after filtering.')

            # Read image and apply transformations
            image = Image.open(frame_path).convert('RGB')
            if self.imgaug_pipeline is not None:
                # expands add batch dim for imgaug
                transformed_image = self.imgaug_pipeline(images=np.expand_dims(image, axis=0))
                # get rid of the batch dim
                transformed_image = transformed_image[0]
            else:
                transformed_image = image

            transformed_image = self.pytorch_transform(transformed_image)
            clip_images.append(transformed_image)
            clip_image_paths.append(str(frame_path))

        # Stack images to create video clip tensor: (16, 3, 224, 224)
        video_clip = torch.stack(clip_images, dim=0)
        assert video_clip.shape == (16, 3, 224, 224), f'Video clip shape is {video_clip.shape}, expected (16, 3, 224, 224)'
        return ExampleDict(
            image=video_clip,  # shape (16, 3, 224, 224)
            video=video_id,
            idx=idx,
            image_path=clip_image_paths,  # list of paths for all 16 frames
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

    def __init__(self, data_dir: str | Path, imgaug_pipeline: Callable | None, config: dict = None) -> None:
        """Initialize a multi-view dataset.

        Parameters
        ----------
        data_dir: absolute path to data directory
        imgaug_transform: imgaug transform pipeline to apply to images
        config: configuration dictionary containing output_mod and 3d_data_dir

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
        
        # Read output_mod from config
        self.config = config or {}
        self.output_mod = self.config.get('data', {}).get('output_mod', ['rgb'])
        if isinstance(self.output_mod, str):
            self.output_mod = [self.output_mod]
        
        # Log output_mod configuration
        logger.info(f'Output modes: {self.output_mod}')

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
        video_id = unique_frame_id.split('/')[0]
        frame_id = unique_frame_id.split('/')[1]

        # random a number between 0 and len(self.available_views)
        input_view_dict = {}
        input_view_paths = []
        for view in self.available_views:
            input_view_path = self.frame_id_to_path[unique_frame_id][view]
            input_view_paths.append(str(input_view_path))
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
        input_image = torch.stack([input_view_dict[view] for view in self.available_views], dim=0) # shape (view, channels, img_height, img_width)
        
        # Build output_image based on output_mod
        output_channels = []
        for mode in self.output_mod:
            if mode == 'rgb':
                # RGB channels from input_image
                output_channels.append(input_image)  # shape (view, 3, H, W)
            elif mode == 'depth':
                # Load depth from 3D data
                depth_data = self._load_3d_data(video_id, frame_id, 'depth')
                output_channels.append(depth_data)  # shape (view, 1, H, W)
            elif mode == 'world_points':
                # Load world_points from 3D data
                world_points_data = self._load_3d_data(video_id, frame_id, 'world_points')
                output_channels.append(world_points_data)  # shape (view, 3, H, W)
            else:
                raise ValueError(f'Unknown output mode: {mode}')
        
        # Concatenate all channels
        if len(output_channels) == 1:
            output_image = output_channels[0]
        else:
            output_image = torch.cat(output_channels, dim=1)  # shape (view, total_channels, H, W)
        return MultiViewDict(
            input_image=input_image,  # shape (view, channels, img_height, img_width)
            output_image=output_image,  # shape (view, channels, img_height, img_width)
            video_id=video_id,
            frame_id=frame_id,
            idx=idx,
            input_view_paths=input_view_paths,
        )
    
    def _load_3d_data(self, video_id: str, frame_id: str, data_type: str) -> torch.Tensor:
        """Load 3D data (depth or world_points) from .npy file.
        
        Parameters
        ----------
        video_id : str
            Video identifier
        frame_id : str
            Frame identifier (e.g., "img_00000001.png")
        data_type : str
            Type of data to load: 'depth' or 'world_points'
        
        Returns
        -------
        torch.Tensor
            Tensor with shape (num_views, channels, height, width)
            - depth: (num_views, 1, 224, 224)
            - world_points: (num_views, 3, 224, 224)
        """
        # Convert frame_id to 3D frame_id format matching create_3d_ssl.py
        # Original create_3d_ssl.py does: frame_id = frame_id.split('.')[0][3:], then frame_id = '3d_'+frame_id
        # So: "img_00000001.png" -> "img_00000001" -> "_00000001" -> "3d__00000001"
        # But this seems odd, let's match the exact logic:
        frame_id_base = frame_id.split('.')[0]  # Remove extension: "img_00000001.png" -> "img_00000001"
        frame_id_3d = '3d_' + frame_id_base[3:]  # Remove first 3 chars and add "3d_" prefix
        
        # Load .npy file
        npy_path = self.data_dir / video_id / f"{frame_id_3d}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f'3D data file not found: {npy_path}')
        
        data_dict = np.load(npy_path, allow_pickle=True).item()
        
        # Check view order if metadata is available
        if 'metadata' in data_dict and 'view_list' in data_dict['metadata']:
            saved_view_list = data_dict['metadata']['view_list']
            if saved_view_list != self.available_views:
                logger.warning(f'View order mismatch in {npy_path}. '
                           f'Saved views: {saved_view_list}, Dataset views: {self.available_views}. '
                           f'Assuming the data is in the correct order based on dataset.available_views.')
        
        # Extract the requested data type
        if data_type not in data_dict:
            raise KeyError(f'Data type {data_type} not found in {npy_path}. Available keys: {list(data_dict.keys())}')
        
        data = data_dict[data_type]  # numpy array
        
        # Convert to torch tensor
        # Handle both numpy array and torch tensor (in case it was saved as tensor)
        if isinstance(data, torch.Tensor):
            data_tensor = data.float()
        else:
            data_tensor = torch.from_numpy(data).float()
        
        # The saved data from create_3d_ssl.py is already in format (S, C, H, W) where:
        # - depth: (S, 1, 224, 224) after permute and interpolate
        # - world_points: (S, 3, 224, 224) after permute and interpolate
        # where S is the number of views
        
        # Verify the tensor has the expected dimensions
        if data_tensor.dim() != 4:
            raise ValueError(f'Expected 4D tensor (S, C, H, W) for {data_type}, got {data_tensor.dim()}D tensor with shape {data_tensor.shape}')
        
        # Verify channel dimension
        expected_channels = 1 if data_type == 'depth' else 3
        if data_tensor.shape[1] != expected_channels:
            raise ValueError(f'Expected {data_type} to have {expected_channels} channels, got {data_tensor.shape[1]} channels')
        
        # Ensure data matches number of views
        num_views = len(self.available_views)
        if data_tensor.shape[0] != num_views:
            raise ValueError(f'Number of views mismatch: expected {num_views}, got {data_tensor.shape[0]}. '
                           f'Check that the view order in the saved .npy file matches dataset.available_views')
        
        # Verify spatial dimensions are 224x224 (should already be from create_3d_ssl.py)
        if data_tensor.shape[2] != 224 or data_tensor.shape[3] != 224:
            logger.warning(f'3D data spatial dimensions are {data_tensor.shape[2]}x{data_tensor.shape[3]}, expected 224x224. Resizing...')
            # Resize if needed (shouldn't be necessary if create_3d_ssl.py worked correctly)
            data_tensor = torch.nn.functional.interpolate(
                data_tensor, size=(224, 224), mode='bilinear', align_corners=False
            )
        
        return data_tensor

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
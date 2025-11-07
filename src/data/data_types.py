from typing import TypedDict, Union

from jaxtyping import Float
from torch import Tensor


class ExampleDict(TypedDict):
    """Return type when calling BaseDataset.__getitem()__."""
    image: Union[
        Float[Tensor, 'channels image_height image_width'],
        Float[Tensor, 'batch channels image_height image_width'],
    ]
    video: Union[str, list[str]]
    idx: Union[int, list[int]]
    image_path: Union[str, list[str]]


class MultiViewDict(TypedDict):
    """Return type when calling MVDataset.__getitem()__ for multi-view learning."""
    input_image: Union[
        Float[Tensor, 'channels image_height image_width'],
        Float[Tensor, 'batch channels image_height image_width'],
        Float[Tensor, 'batch view channels image_height image_width']
    ]
    output_image: Union[
        Float[Tensor, 'channels image_height image_width'],
        Float[Tensor, 'batch channels image_height image_width'],
    ]
    video_id: Union[str, list[str]]
    frame_id: Union[str, list[str]]
    idx: Union[int, list[int]]
    input_image_path: Union[str, list[str]]
    output_image_path: Union[str, list[str]]
    input_view: Union[str, list[str]]
    output_view: Union[str, list[str]]

class EncodingDict(TypedDict):
    """
    Return type for encoding multi-view data.

    - input_video_view: Dict[str, Float[Tensor, 'batch time channels image_height image_width']]
    - input_keypoints_view: Dict[str, Float[Tensor, 'batch time num_keypoints']]
    - input_discrete_keypoints_view: Dict[str, Dict[str, Float[Tensor, 'batch time']]]
    - spike: Float[Tensor, 'batch time neurons']
    """
    input_video_view: Union[
        dict[str, Float[Tensor, 'batch time channels image_height image_width']],
        dict[str, Float[Tensor, 'time channels image_height image_width']],
    ]
    input_keypoints_view: Union[
        dict[str, Float[Tensor, 'batch time num_keypoints']],
        dict[str, Float[Tensor, 'time num_keypoints']],
    ]
    input_discrete_keypoints_view: Union[
        dict[str, dict[str, Float[Tensor, 'batch time']]],
        dict[str, dict[str, Float[Tensor, 'time']]],
    ]
    spike: Union[
        Float[Tensor, 'batch time neurons'],
        Float[Tensor, 'time neurons'],
    ]

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
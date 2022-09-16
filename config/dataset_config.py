from pydantic import Field
from typing import Optional, List

from config.base_config import BaseConfig


class DatasetConfig(BaseConfig):
    """ Dataset transformations configuration """

    random_rotation: Optional[int] = Field(description="Random rotation coefficient")
    random_horizontal_flip: Optional[float] = Field(description="Random horizontal flip coefficient")
    crop_size: Optional[int] = Field(description="Image crop size")
    crop_padding: Optional[int] = Field(description="Image crop padding")
    means: Optional[List[float]] = Field(description="Normalization means")
    stds: Optional[List[float]] = Field(description="Normalization standard deviations")

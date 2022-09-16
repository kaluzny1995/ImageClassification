import json

from pydantic import BaseModel, Field, constr, root_validator, validator
from typing import Optional

from config.base_config import BaseConfig
from config.paths_config import PathsConfig
from models.e_device_type import EDeviceType


class MainConfig(BaseConfig):
    """ Main configuration - common for all models """

    paths: PathsConfig = Field(description="Configuration for paths")
    random_seed: int = Field(description="Seed for randomizing")
    cuda_device: EDeviceType = Field(description="Name of cuda device")
    non_cuda_device: EDeviceType = Field(description="Name of non-cuda device")
    tt_split_ratio: float = Field(description="Ratio of train-test datasets splitting")
    tv_split_ratio: float = Field(description="Ratio of train-validation datasets splitting")
    is_visualization_saved: bool = Field(description="Are visualizations persisted to storage folder")
    is_visualization_shown: bool = Field(description="Are visualizations shown")
    is_launched_in_notebook: bool = Field(description="Is solution launched in Jupyter Notebook")

    @staticmethod
    def from_json() -> 'MainConfig':
        """ Returns config object based on config.json file """
        config_dict = json.load(open("config.json"))
        return MainConfig(**config_dict['main'])

from pydantic import Field
from typing import Optional

from models.e_dataset import EDataset
from config.base_config import BaseConfig
from config.dataset_config import DatasetConfig
from config.lr_finder_config import LRFinderConfig
from config.model_params_config import ModelParamsConfig
from config.model_hparams_config import ModelHParamsConfig


class ModelBaseConfig(BaseConfig):
    """ Model base configuration - common for all models """

    utilized_dataset: EDataset = Field(description="Dataset for model training and testing")
    ds_param: Optional[DatasetConfig] = Field(description="Utilized dataset parameters config")
    lrf: Optional[LRFinderConfig] = Field(description="Learning rate finder config")
    param: ModelParamsConfig = Field(description="Model parameters config")
    hparam: ModelHParamsConfig = Field(description="Model hyperparameters config")

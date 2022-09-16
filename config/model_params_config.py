from pydantic import Field
from typing import Optional, Union, List, Tuple, Dict, Any

from config.base_config import BaseConfig


class FeatureExtractorConfig(BaseConfig):
    """ Model feature extractors parameters configuration """
    in_out_channels: List[Union[List[int], Tuple[int]]] = Field(description="Convolutional layers channels")
    kernel_size: int = Field(description="Convolutional layers kernel size")
    pool_kernel_size: int = Field(description="Pooling layers kernel size")
    stride: Optional[int] = Field(description="Convolutional layers stride")
    padding: Optional[int] = Field(description="Convolutional layers padding")
    is_batchnorm_used: Optional[bool] = Field(description="Are batch normalization layers used in model")


class ClassifierConfig(BaseConfig):
    """ Model classifiers parameters configuration """
    dims: List[Union[List[int], Tuple[int]]] = Field(description="Dense layers dimensions")
    dropout: Optional[float] = Field(description="Dense layers dropout")


class ModelParamsConfig(BaseConfig):
    """ Model parameters configuration """
    preset: Optional[Dict[str, Any]] = Field(description="Preset layers configuration")
    ft: Optional[FeatureExtractorConfig] = Field(description="Features extractor configuration")
    avg_pool_size: Optional[int] = Field(description="Average pooling size between features and classifier modules")
    clf: Optional[ClassifierConfig] = Field(description="Classifier configuration")

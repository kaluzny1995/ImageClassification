from pydantic import Field
from typing import Optional

from models.e_device_type import EDeviceType
from models.e_criterion import ECriterion
from models.e_optimizer import EOptimizer
from config.base_config import BaseConfig


class LRFinderConfig(BaseConfig):
    """ Model lr finder configuration """
    device: Optional[EDeviceType] = Field(description="Used device type ('cuda' or 'cpu')")
    criterion: ECriterion = Field(description="Criterion (loss) function")
    optimizer: EOptimizer = Field(description="Optimizer function")
    start_lr: float = Field(description="Model start learning rate")
    end_lr: float = Field(description="Model end learning rate")
    num_iter: int = Field(description="Maximum number of finding iterations")

from pydantic import Field
from typing import Optional

from models.e_device_type import EDeviceType
from models.e_criterion import ECriterion
from models.e_optimizer import EOptimizer
from models.e_lr_scheduler import ELRScheduler
from config.base_config import BaseConfig


class ModelHParamsConfig(BaseConfig):
    """ Model hyperparameters configuration """
    device: Optional[EDeviceType] = Field(description="Used device type ('cuda' or 'cpu')")
    batch_size: int = Field(description="Size of the data batches")
    criterion: ECriterion = Field(description="Criterion (loss) function")
    optimizer: EOptimizer = Field(description="Optimizer function")
    lr_scheduler: Optional[ELRScheduler] = Field(description="Learning rate scheduler function")
    learning_rate: float = Field(description="Model learning rate")
    epochs: int = Field(description="Number of training epochs")

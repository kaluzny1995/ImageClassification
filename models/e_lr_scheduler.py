import torch.optim.lr_scheduler as lr_scheduler

from enum import Enum


class ELRScheduler(Enum):
    ONE_CYCLE_LR = lr_scheduler.OneCycleLR

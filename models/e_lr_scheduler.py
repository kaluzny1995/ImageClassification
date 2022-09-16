from enum import Enum


class ELRScheduler(str, Enum):
    ONE_CYCLE_LR = "one_cycle_lr"

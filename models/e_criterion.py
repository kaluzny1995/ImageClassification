import torch.nn as nn

from enum import Enum


class ECriterion(Enum):
    CE = nn.CrossEntropyLoss
    BCE = nn.BCELoss
    BCEL = nn.BCEWithLogitsLoss

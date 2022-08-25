import torch.optim as optim

from enum import Enum


class EOptimizer(Enum):
    ADAM = optim.Adam
    ADAMAX = optim.Adamax
    ADAGRAD = optim.Adagrad

from enum import Enum


class EDataset(str, Enum):
    MNIST = "mnist"
    CIFAR10 = "cifar10"
    CUB200 = "cub200"

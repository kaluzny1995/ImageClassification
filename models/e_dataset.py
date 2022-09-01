from enum import Enum

from datasets.mnist_dataset import MNISTDataset
from datasets.cifar10_dataset import CIFAR10Dataset
from datasets.cub200_dataset import CUB200Dataset


class EDataset(Enum):
    MNIST = MNISTDataset
    CIFAR10 = CIFAR10Dataset
    CUB200 = CUB200Dataset

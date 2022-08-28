from enum import Enum

from datasets.mnist_dataset import MNISTDataset
from datasets.cifar10_dataset import CIFAR10Dataset


class EDataset(Enum):
    MNIST = MNISTDataset
    CIFAR10 = CIFAR10Dataset

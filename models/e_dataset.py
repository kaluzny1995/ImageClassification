from enum import Enum

from datasets.mnist_dataset import MNISTDataset


class EDataset(Enum):
    MNIST = MNISTDataset

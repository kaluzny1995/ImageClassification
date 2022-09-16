from datasets.mnist_dataset import MNISTDataset
from datasets.cifar10_dataset import CIFAR10Dataset
from datasets.cub200_dataset import CUB200Dataset


datasets_dict = dict(
    mnist=MNISTDataset,
    cifar10=CIFAR10Dataset,
    cub200=CUB200Dataset
)

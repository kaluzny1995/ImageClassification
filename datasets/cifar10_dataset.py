import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from datasets.i_dataset import IDataset
from utils.util import create_dir_if_not_exists


class CIFAR10Dataset(IDataset):
    def __init__(self, dataset_path, validation_ratio):
        super(CIFAR10Dataset, self).__init__()
        self.dataset_path = dataset_path
        self.validation_ratio = validation_ratio

        create_dir_if_not_exists(self.dataset_path)
        train_data = datasets.CIFAR10(root=dataset_path, train=True, download=True)

        means = train_data.data.mean(axis=(0, 1, 2)) / 255
        stds = train_data.data.std(axis=(0, 1, 2)) / 255

        self.train_transforms = transforms.Compose([
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop(32, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds)
        ])
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds)
        ])

        self.train_data = datasets.CIFAR10(dataset_path, train=True, download=True, transform=self.train_transforms)
        self.test_data = datasets.CIFAR10(dataset_path, train=False, download=True, transform=self.test_transforms)

    def get_datasets(self):
        train_data, valid_data = data.random_split(self.train_data, [
            int(len(self.train_data) * self.validation_ratio),
            len(self.train_data) - int(len(self.train_data) * self.validation_ratio)
        ])
        setattr(valid_data.dataset, "transform", self.test_transforms)

        return train_data, valid_data, self.test_data

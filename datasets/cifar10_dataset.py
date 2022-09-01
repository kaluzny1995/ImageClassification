import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import copy

from datasets.i_dataset import IDataset
from utils.util import create_dir_if_not_exists


class CIFAR10Dataset(IDataset):
    def __init__(self, data_path, validation_ratio, are_parameters_calculated=True, **params):
        super(CIFAR10Dataset, self).__init__()
        self.data_path = data_path
        self.validation_ratio = validation_ratio

        create_dir_if_not_exists(self.data_path)

        random_rotation = params.get("random_rotation", 5)
        random_horizontal_flip = params.get("random_horizontal_flip", 0.5)
        crop_size = params.get("crop_size", 32)
        crop_padding = params.get("crop_padding", 2)
        if are_parameters_calculated:
            train_data = datasets.CIFAR10(root=data_path, train=True, download=True)
            means = train_data.data.mean(axis=(0, 1, 2)) / 255
            stds = train_data.data.std(axis=(0, 1, 2)) / 255
        else:
            means = params.get("means", [0.485, 0.456, 0.406])
            stds = params.get("stds", [0.229, 0.224, 0.225])

        self.train_transforms = transforms.Compose([
            transforms.RandomRotation(random_rotation),
            transforms.RandomHorizontalFlip(random_horizontal_flip),
            transforms.RandomCrop(crop_size, padding=crop_padding),
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds)
        ])
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds)
        ])

        self.train_data = datasets.CIFAR10(data_path, train=True, download=True, transform=self.train_transforms)
        self.test_data = datasets.CIFAR10(data_path, train=False, download=True, transform=self.test_transforms)

    def get_datasets(self):
        train_data, valid_data = data.random_split(self.train_data, [
            int(len(self.train_data) * self.validation_ratio),
            len(self.train_data) - int(len(self.train_data) * self.validation_ratio)
        ])
        valid_data = copy.deepcopy(valid_data)
        setattr(valid_data.dataset, "transform", self.test_transforms)

        return train_data, valid_data, self.test_data

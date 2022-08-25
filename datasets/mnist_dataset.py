import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from datasets.i_dataset import IDataset
from utils.util import create_dir_if_not_exists


class MNISTDataset(IDataset):
    def __init__(self, dataset_path, validation_ratio):
        super(MNISTDataset, self).__init__()
        self.dataset_path = dataset_path
        self.validation_ratio = validation_ratio

        create_dir_if_not_exists(self.dataset_path)
        train_data = datasets.MNIST(root=dataset_path, train=True, download=True)
        mean = train_data.data.float().mean() / 255
        std = train_data.data.float().std() / 255

        self.train_transforms = transforms.Compose([
            transforms.RandomRotation(5, fill=(0,)),
            transforms.RandomCrop(28, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std])
        ])
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std])
        ])

        self.train_data = datasets.MNIST(root=dataset_path, train=True, download=True, transform=self.train_transforms)
        self.test_data = datasets.MNIST(root=dataset_path, train=False, download=True, transform=self.test_transforms)

    def get_datasets(self):
        train_data, valid_data = data.random_split(self.train_data, [
            int(len(self.train_data) * self.validation_ratio),
            len(self.train_data) - int(len(self.train_data) * self.validation_ratio)
        ])
        setattr(valid_data.dataset, "transform", self.test_transforms)

        return train_data, valid_data, self.test_data

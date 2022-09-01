import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.datasets.utils as datasets_utils
import torchvision.transforms as transforms

import os
import subprocess
import shutil
import copy

from config.kaggle_config import KaggleConfig
from datasets.i_dataset import IDataset
from utils.util import create_dir_if_not_exists
from utils.calculation import calculate_means_and_stds


class CUB200Dataset(IDataset):
    def __init__(self, data_path, test_ratio, validation_ratio, are_parameters_calculated=True, **params):
        super(CUB200Dataset, self).__init__()
        self.data_path = data_path
        self.dataset_path = os.path.join(data_path, "CUB200")
        self.images_path = os.path.join(self.dataset_path, "CUB_200_2011", "images")
        self.train_path = os.path.join(self.dataset_path, "train")
        self.test_path = os.path.join(self.dataset_path, "test")

        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio

        create_dir_if_not_exists(self.data_path)
        if not os.path.exists(self.dataset_path):
            create_dir_if_not_exists(self.dataset_path)
            self.__download()
            self.__build_datasets()
            shutil.rmtree(os.path.join(self.dataset_path, "CUB_200_2011"))

        random_rotation = params.get("random_rotation", 5)
        random_horizontal_flip = params.get("random_horizontal_flip", 0.5)
        crop_size = params.get("crop_size", 32)
        crop_padding = params.get("crop_padding", 2)
        if are_parameters_calculated:
            train_data = datasets.ImageFolder(root=self.train_path, transform=transforms.ToTensor())
            means, stds = calculate_means_and_stds(train_data)
        else:
            means = params.get("means", [0.485, 0.456, 0.406])
            stds = params.get("stds", [0.229, 0.224, 0.225])

        self.train_transforms = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomRotation(random_rotation),
            transforms.RandomHorizontalFlip(random_horizontal_flip),
            transforms.RandomCrop(crop_size, padding=crop_padding),
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds)
        ])
        self.test_transforms = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds)
        ])

        self.train_data = datasets.ImageFolder(root=self.train_path, transform=self.train_transforms)
        setattr(self.train_data, 'classes',
                list(map(lambda c: CUB200Dataset.__format_label(c), self.train_data.classes)))
        self.test_data = datasets.ImageFolder(root=self.test_path, transform=self.test_transforms)
        setattr(self.test_data, 'classes',
                list(map(lambda c: CUB200Dataset.__format_label(c), self.test_data.classes)))

    @staticmethod
    def __format_label(label):
        label = label.split('.')[-1]
        label = label.replace('_', ' ')
        label = label.title()
        label = label.replace(' ', '')
        return label

    def __download(self):
        # WARNING: KAGGLE_USERNAME and KAGGLE_KEY environment variables must be set before launching
        # or place your own kaggle.json file into the project main directory
        # Instructions here: https://github.com/Kaggle/kaggle-api#api-credentials
        kaggle_config = KaggleConfig.from_json()
        os.environ['KAGGLE_USERNAME'] = kaggle_config.username
        os.environ['KAGGLE_KEY'] = kaggle_config.key

        subprocess.run("kaggle datasets download veeralakrishna/200-bird-species-with-11788-images --unzip", shell=True)
        datasets_utils.extract_archive('CUB_200_2011.tgz', self.dataset_path)
        subprocess.run("rm CUB_200_2011.tgz", shell=True)
        subprocess.run("rm segmentations.tgz", shell=True)

    def __build_datasets(self):
        os.makedirs(self.train_path)
        os.makedirs(self.test_path)

        classes = os.listdir(self.images_path)
        for c in classes:
            class_path = os.path.join(self.images_path, c)
            images = os.listdir(class_path)

            n_train = int(len(images) * self.test_ratio)
            train_images = images[:n_train]
            test_images = images[n_train:]

            os.makedirs(os.path.join(self.train_path, c), exist_ok=True)
            os.makedirs(os.path.join(self.test_path, c), exist_ok=True)

            for image in train_images:
                image_src = os.path.join(class_path, image)
                image_dst = os.path.join(self.train_path, c, image)
                shutil.copyfile(image_src, image_dst)

            for image in test_images:
                image_src = os.path.join(class_path, image)
                image_dst = os.path.join(self.test_path, c, image)
                shutil.copyfile(image_src, image_dst)

    def get_datasets(self):
        train_data, valid_data = data.random_split(self.train_data, [
            int(len(self.train_data) * self.validation_ratio),
            len(self.train_data) - int(len(self.train_data) * self.validation_ratio)
        ])
        valid_data = copy.deepcopy(valid_data)
        setattr(valid_data.dataset, "transform", self.test_transforms)

        return train_data, valid_data, self.test_data

import torch
import torch.nn as nn

from enum import Enum

from models.dict_pretrained_model import pretrained_models_dict
from utils.util import create_dir_if_not_exists


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, is_downsampled=False):
        super().__init__()
        self.is_downsampled = is_downsampled

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if self.is_downsampled:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = None

    def forward(self, x):
        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.is_downsampled:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, is_downsampled=False):
        super().__init__()
        self.is_downsampled = is_downsampled

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.relu = nn.ReLU(inplace=True)

        if self.is_downsampled:
            conv = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(self.expansion * out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = None

    def forward(self, x):
        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.is_downsampled:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x


class EResNetBlock(Enum):
    basic = BasicBlock
    bottleneck = Bottleneck


class ResNet(nn.Module):
    def __init__(self, preset, output_dim,
                 save_path, model_name):
        super().__init__()
        if model_name not in preset.keys():
            raise ValueError(f"No preset found for model '{model_name}'. Provide one of: {preset.keys()}")

        self.preset = preset
        self.output_dim = output_dim

        block = EResNetBlock[preset[model_name]['block']].value
        n_blocks = preset[model_name]['n_blocks']
        channels = preset[model_name]['channels']
        assert len(n_blocks) == len(channels) == 4
        self.in_channels = channels[0]

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.__get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.__get_resnet_layer(block, n_blocks[1], channels[1], stride=2)
        self.layer3 = self.__get_resnet_layer(block, n_blocks[2], channels[2], stride=2)
        self.layer4 = self.__get_resnet_layer(block, n_blocks[3], channels[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, output_dim)

        self.save_path = save_path
        self.model_name = model_name
        create_dir_if_not_exists(f"{self.save_path}/{self.model_name}")
        self.__init_model()

    def __get_model_path(self, name):
        return f"{self.save_path}/{self.model_name}/{name}.pt"

    def __get_resnet_layer(self, block, n_blocks, channels, stride=1):
        layers = []

        is_downsampled = self.in_channels != block.expansion * channels
        layers.append(block(self.in_channels, channels, stride, is_downsampled))

        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels

        return nn.Sequential(*layers)

    def __init_model(self):
        pretrained_model_class = pretrained_models_dict[self.model_name]
        pretrained_model = pretrained_model_class(pretrained=True)
        input_dim = pretrained_model.fc.in_features

        final_clf = nn.Linear(input_dim, self.output_dim)
        pretrained_model.fc = final_clf
        self.load_state_dict(pretrained_model.state_dict())

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)

        return x, h

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self):
        torch.save(self.state_dict(), self.__get_model_path("model"))

    def load(self):
        self.load_state_dict(torch.load(self.__get_model_path("model")))

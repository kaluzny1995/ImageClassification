import torch
import torch.nn as nn

from models.dict_pretrained_model import pretrained_models_dict
from utils.util import create_dir_if_not_exists


class VGG(nn.Module):
    def __init__(self, preset,
                 in_channels, kernel_size, pool_kernel_size, padding, is_batchnorm_used,
                 avg_pool_size,
                 dims, dropout,
                 save_path, model_name):
        super().__init__()
        if model_name not in preset.keys():
            raise ValueError(f"No preset found for model '{model_name}'. Provide one of: {preset.keys()}")

        self.preset = preset
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.padding = padding
        self.is_batchnorm_used = is_batchnorm_used
        self.avg_pool_size = avg_pool_size
        self.dims = dims
        self.dropout = dropout

        self.features = self.__get_vgg_layers(self.preset[model_name], self.is_batchnorm_used)

        self.avgpool = nn.AdaptiveAvgPool2d(self.avg_pool_size)

        self.classifier = nn.Sequential(
            nn.Linear(*self.dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(*self.dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(*self.dims[-1]),
        )

        self.save_path = save_path
        self.model_name = model_name
        create_dir_if_not_exists(f"{self.save_path}/{self.model_name}")
        self.__init_model()

    def __get_model_path(self, name):
        return f"{self.save_path}/{self.model_name}/{name}.pt"

    def __get_vgg_layers(self, sequence, is_batchnorm_used=True):
        layers = list([])
        in_channels = self.in_channels

        for element in sequence:
            assert element == 'M' or isinstance(element, int)
            if element == 'M':
                layers += [nn.MaxPool2d(kernel_size=self.pool_kernel_size)]
            else:
                conv2d = nn.Conv2d(in_channels, element, kernel_size=self.kernel_size, padding=self.padding)
                if is_batchnorm_used:
                    layers += [conv2d, nn.BatchNorm2d(element), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = element

        return nn.Sequential(*layers)

    def __init_model(self):
        pretrained_model_class = pretrained_models_dict[self.model_name] \
            if not self.is_batchnorm_used else pretrained_models_dict[self.model_name + "_bn"]
        pretrained_model = pretrained_model_class(pretrained=True)
        in_dim = pretrained_model.classifier[-1].in_features

        final_clf = nn.Linear(in_dim, self.dims[-1][-1])
        pretrained_model.classifier[-1] = final_clf
        self.load_state_dict(pretrained_model.state_dict())

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self):
        torch.save(self.state_dict(), self.__get_model_path("model"))

    def load(self):
        self.load_state_dict(torch.load(self.__get_model_path("model")))

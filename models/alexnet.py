import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import create_dir_if_not_exists


class AlexNet(nn.Module):
    def __init__(self, in_out_channels, kernel_size, pool_kernel_size, stride, padding,
                 dims, dropout,
                 save_path, model_name):
        """
        AlexNet - convolutional neural network model
        :param in_out_channels: Input/output channels for each conv. layer
        :type in_out_channels: List[Tuple[int, int]]
        :param kernel_size: Conv. layers kernel size
        :type kernel_size: int
        :param pool_kernel_size: Pooling layers kernel size
        :type pool_kernel_size: int
        :param stride: Conv. layers stride
        :type stride: int
        :param padding: Conv. layers padding
        :type padding: int
        :param dims: Input/output dimensions for each dense layer
        :type dims: List[Tuple[int, int]]
        :param dropout: Dense layers dropout
        :type dropout: float
        :param save_path: Base path for model saving
        :type save_path: str
        :param model_name: Name of the model
        :type model_name: str
        """
        super().__init__()

        self.in_out_channels = in_out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.stride = stride
        self.padding = padding
        self.dims = dims
        self.dropout = dropout

        self.features = nn.Sequential(
            nn.Conv2d(*self.in_out_channels[0], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.MaxPool2d(self.pool_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(*self.in_out_channels[1], kernel_size=self.kernel_size, padding=self.padding),
            nn.MaxPool2d(self.pool_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(*self.in_out_channels[2], kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(*self.in_out_channels[3], kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(*self.in_out_channels[4], kernel_size=self.kernel_size, padding=self.padding),
            nn.MaxPool2d(self.pool_kernel_size),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(*self.dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(*self.dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(*self.dims[2]),
        )

        self.apply(AlexNet.__init_params)

        self.save_path = save_path
        self.model_name = model_name
        create_dir_if_not_exists(f"{self.save_path}/{self.model_name}")

    def __get_model_path(self, name):
        return f"{self.save_path}/{self.model_name}/{name}.pt"

    @staticmethod
    def __init_params(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self):
        torch.save(self.state_dict(), self.__get_model_path("model"))

    def load(self):
        self.load_state_dict(torch.load(self.__get_model_path("model")))

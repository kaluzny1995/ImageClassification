import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import create_dir_if_not_exists


class LeNet(nn.Module):
    def __init__(self, in_out_channels, kernel_size, pool_kernel_size,
                 dims,
                 save_path, model_name):
        """
        LeNet - convolutional neural network model
        :param in_out_channels: Convolutional layers channels
        :type in_out_channels: List[List[int]]
        :param kernel_size: Convolutional layer kernel size
        :type kernel_size: int
        :param pool_kernel_size: Pooling layer kernel size
        :type pool_kernel_size: int
        :param dims: Dense layers dimensions
        :type dims: List[List[int]]
        :param save_path: Base path for model saving
        :type save_path: str
        :param model_name: Name of the model
        :type model_name: str
        """
        super().__init__()

        self.in_out_channels = in_out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.dims = dims

        self.conv1 = nn.Conv2d(*in_out_channels[0], kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(*in_out_channels[-1], kernel_size=kernel_size)

        self.fc_1 = nn.Linear(*dims[0])
        self.fc_2 = nn.Linear(*dims[1])
        self.fc_3 = nn.Linear(*dims[-1])

        self.save_path = save_path
        self.model_name = model_name
        create_dir_if_not_exists(f"{self.save_path}/{self.model_name}")

    def __get_model_path(self, name):
        return f"{self.save_path}/{self.model_name}/{name}.pt"

    def forward(self, x):
        # x = [batch size, 1, 28, 28]
        x = self.conv1(x)
        # x = [batch size, 6, 24, 24]
        x = F.max_pool2d(x, kernel_size=self.pool_kernel_size)
        # x = [batch size, 6, 12, 12]
        x = F.relu(x)
        x = self.conv2(x)
        # x = [batch size, 16, 8, 8]
        x = F.max_pool2d(x, kernel_size=self.pool_kernel_size)
        # x = [batch size, 16, 4, 4]
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        # x = [batch size, 16*4*4 = 256]
        h = x
        x = self.fc_1(x)
        # x = [batch size, 120]
        x = F.relu(x)
        x = self.fc_2(x)
        # x = batch size, 84]
        x = F.relu(x)
        x = self.fc_3(x)
        # x = [batch size, output dim]
        return x, h

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self):
        torch.save(self.state_dict(), self.__get_model_path("model"))

    def load(self):
        self.load_state_dict(torch.load(self.__get_model_path("model")))

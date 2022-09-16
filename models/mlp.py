import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import create_dir_if_not_exists


class MLP(nn.Module):
    def __init__(self, dims, save_path, model_name):
        """
        MLP - dense neural network model
        :param dims: Dense layers dimensions
        :type dims: List[Tuple[int, int]]
        :param save_path: Base path for model saving
        :type save_path: str
        :param model_name: Name of the model
        :type model_name: str
        """
        super().__init__()

        self.input_fc = nn.Linear(*dims[0])
        self.hidden_fc = nn.Linear(*dims[1])
        self.output_fc = nn.Linear(*dims[-1])

        self.save_path = save_path
        self.model_name = model_name
        create_dir_if_not_exists(f"{self.save_path}/{self.model_name}")

    def __get_model_path(self, name):
        return f"{self.save_path}/{self.model_name}/{name}.pt"

    def forward(self, x):
        # x = [batch size, height, width]
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        # x = [batch size, height * width]
        h_1 = F.relu(self.input_fc(x))
        # h_1 = [batch size, 250]
        h_2 = F.relu(self.hidden_fc(h_1))
        # h_2 = [batch size, 100]
        y_pred = self.output_fc(h_2)
        # y_pred = [batch size, output dim]
        return y_pred, h_2

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self):
        torch.save(self.state_dict(), self.__get_model_path("model"))

    def load(self):
        self.load_state_dict(torch.load(self.__get_model_path("model")))

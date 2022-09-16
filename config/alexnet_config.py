import json

from models.e_dataset import EDataset
from models.e_device_type import EDeviceType
from models.e_criterion import ECriterion
from models.e_optimizer import EOptimizer
from config.model_base_config import ModelBaseConfig


class AlexNetConfig(ModelBaseConfig):
    
    @staticmethod
    def from_json() -> 'AlexNetConfig':
        """ Returns config object based on config.json file """
        config_dict = json.load(open("config.json"))
        return AlexNetConfig(**config_dict['alexnet'])

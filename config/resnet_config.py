import json

from config.model_base_config import ModelBaseConfig


class ResNetConfig(ModelBaseConfig):

    @staticmethod
    def from_json() -> 'ResNetConfig':
        """ Returns config object based on config.json file """
        config_dict = json.load(open("config.json"))
        return ResNetConfig(**config_dict['resnet'])

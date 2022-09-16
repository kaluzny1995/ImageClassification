import json

from config.model_base_config import ModelBaseConfig


class VGGConfig(ModelBaseConfig):

    @staticmethod
    def from_json() -> 'VGGConfig':
        """ Returns config object based on config.json file """
        config_dict = json.load(open("config.json"))
        return VGGConfig(**config_dict['vgg'])

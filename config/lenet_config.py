import json

from config.model_base_config import ModelBaseConfig


class LeNetConfig(ModelBaseConfig):

    @staticmethod
    def from_json() -> 'LeNetConfig':
        """ Returns config object based on config.json file """
        config_dict = json.load(open("config.json"))
        return LeNetConfig(**config_dict['lenet'])

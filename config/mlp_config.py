import json

from config.model_base_config import ModelBaseConfig


class MLPConfig(ModelBaseConfig):

    @staticmethod
    def from_json() -> 'MLPConfig':
        """ Returns config object based on config.json file """
        config_dict = json.load(open("config.json"))
        return MLPConfig(**config_dict['mlp'])

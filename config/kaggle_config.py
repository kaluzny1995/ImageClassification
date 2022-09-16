from pydantic import Field

import json

from config.base_config import BaseConfig


class KaggleConfig(BaseConfig):
    username: str = Field(description="Kaggle username")
    key: str = Field(description="Kaggle API token")

    @staticmethod
    def from_json() -> 'KaggleConfig':
        """ Returns config object based on config.json file """
        config_dict = json.load(open("kaggle.json"))
        return KaggleConfig(**config_dict)

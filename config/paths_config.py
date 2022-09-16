import json

from pydantic import BaseModel, Field
from typing import Optional

from config.base_config import BaseConfig


class PathsConfig(BaseConfig):
    """ Paths configuration - paths for data loading and models/visualizations persistence """

    root: str = Field(description="Path of the root folder")
    data: str = Field(description="Path of the data folder")
    visualizations: str = Field(description="Path of visualizations storage folder")
    models: str = Field(description="Path of models storage folder")

    @staticmethod
    def from_json() -> 'PathsConfig':
        """ Returns config object based on config.json file """
        config_dict = json.load(open("config.json"))
        return PathsConfig(**config_dict['main']['paths'])

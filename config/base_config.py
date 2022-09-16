import json

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class BaseConfig(BaseModel):
    """ Base configuration - common for all configurations """

    name: Optional[str] = Field(description="Configuration name")
    description: Optional[str] = Field(description="Configuration description")

    class Config:
        use_enum_values = True

    @staticmethod
    def from_json() -> 'BaseConfig':
        """ Returns config object based on config.json file """
        return BaseConfig()

    def to_dict(self) -> Dict[str, Any]:
        """ Returns dictionary of config parameters """
        config_dict = dict(map(lambda x: (x, getattr(self, x)),
                               filter(lambda x: not x.startswith('_') and not callable(getattr(self, x)),
                                      dir(self))))
        return config_dict

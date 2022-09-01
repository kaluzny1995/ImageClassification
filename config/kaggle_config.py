import json


class KaggleConfig:
    def __init__(self, username, key):
        """
        Kaggle credentials config
        :param username: Kaggle username
        :type username: str
        :param key: Kaggle API token
        :type key: str
        """
        self.username = username
        self.key = key

    @staticmethod
    def from_json():
        """ Returns config object based on config.json file """
        config_dict = json.load(open("kaggle.json"))
        return KaggleConfig(config_dict['username'], config_dict['key'])

    def to_dict(self):
        """ Returns dictionary of config parameters """
        config_dict = dict(map(lambda x: (x, getattr(self, x)),
                               filter(lambda x: not x.startswith('_') and not callable(getattr(self, x)),
                                      dir(self))))
        return config_dict

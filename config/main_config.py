import json


from models.e_device_type import EDeviceType


class MainConfig:
    def __init__(self, name, description,
                 path_root, path_data, path_storage_visualization, path_storage_models,
                 random_seed, cuda_device, non_cuda_device, tt_split_ratio, tv_split_ratio,
                 is_visualization_saved, is_visualization_shown, is_launched_in_notebook):
        """
        Main configuration - common for all models
        :param name: Configuration name
        :type name: str
        :param description: Configuration description
        :type description: str
        :param path_root: Path of the root folder
        :type path_root: str
        :param path_data: Path of the data folder
        :type path_data: str
        :param path_storage_visualization: Path of visualizations storage folder
        :type path_storage_visualization: str
        :param path_storage_models: Path of models storage folder
        :type path_storage_models: str
        :param random_seed: Seed for randomizing
        :type random_seed: int
        :param cuda_device: Name of cuda device
        :type cuda_device: EDeviceType
        :param non_cuda_device: Name of non-cuda device
        :type non_cuda_device: EDeviceType
        :param tt_split_ratio: Ratio of train-test datasets splitting
        :type tt_split_ratio: float
        :param tv_split_ratio: Ratio of train-validation datasets splitting
        :type tv_split_ratio: float
        :param is_visualization_saved: Are visualizations persisted to storage folder
        :type is_visualization_saved: bool
        :param is_visualization_shown: Are visualizations shown
        :type is_visualization_shown: bool
        :param is_launched_in_notebook: Is solution launched in Jupyter Notebook
        :type is_launched_in_notebook: bool
        """
        self.name = name
        self.description = description
        self.path_root = path_root
        self.path_data = path_data
        self.path_storage_visualization = path_storage_visualization
        self.path_storage_models = path_storage_models
        self.random_seed = random_seed
        self.cuda_device = cuda_device
        self.non_cuda_device = non_cuda_device
        self.tt_split_ratio = tt_split_ratio
        self.tv_split_ratio = tv_split_ratio
        self.is_visualization_saved = is_visualization_saved
        self.is_visualization_shown = is_visualization_shown
        self.is_launched_in_notebook = is_launched_in_notebook

    @staticmethod
    def from_json():
        """ Returns config object based on config.json file """
        config_dict = json.load(open("config.json"))
        return MainConfig(config_dict['main']['name'],
                          config_dict['main']['description'],
                          config_dict['main']['path']['root'],
                          config_dict['main']['path']['data'],
                          config_dict['main']['path']['storage']['visualizations'],
                          config_dict['main']['path']['storage']['models'],
                          config_dict['main']['random_seed'],
                          EDeviceType[config_dict['main']['cuda_device']],
                          EDeviceType[config_dict['main']['non_cuda_device']],
                          config_dict['main']['tt_split_ratio'],
                          config_dict['main']['tv_split_ratio'],
                          config_dict['main']['is_visualization_saved'],
                          config_dict['main']['is_visualization_shown'],
                          config_dict['main']['is_launched_in_notebook'])

    def to_dict(self):
        """ Returns dictionary of config parameters """
        config_dict = dict(map(lambda x: (x, getattr(self, x)),
                               filter(lambda x: not x.startswith('_') and not callable(getattr(self, x)),
                                      dir(self))))
        return config_dict

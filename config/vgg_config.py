import json

from models.e_dataset import EDataset
from models.e_device_type import EDeviceType
from models.e_criterion import ECriterion
from models.e_optimizer import EOptimizer


class VGGConfig:
    def __init__(self, name, description, utilized_dataset,
                 ds_param_random_rotation, ds_param_random_horizontal_flip, ds_param_crop_size, ds_param_crop_padding,
                 ds_param_means, ds_param_stds,
                 lrf_device, lrf_criterion, lrf_optimizer, lrf_start_lr, lrf_end_lr, lrf_num_iter,
                 param_preset,
                 param_ft_in_channels, param_ft_kernel_size, param_ft_pool_kernel_size,
                 param_ft_padding, param_ft_is_batchnorm_used, param_avg_pool_size, param_clf_dims, param_clf_dropout,
                 hparam_device, hparam_batch_size, hparam_criterion,
                 hparam_optimizer, hparam_learning_rate, hparam_epochs):
        """
        VGG NN configuration
        :param name: Name of the model
        :type name: str
        :param description: Model description
        :type description: str
        :param utilized_dataset: Dataset for model training and testing
        :type utilized_dataset: EDataset
        :param ds_param_random_rotation: Dataset transform - random rotation coefficient
        :type ds_param_random_rotation: float
        :param ds_param_random_horizontal_flip: Dataset transform - random horizontal flip coefficient
        :type ds_param_random_horizontal_flip: float
        :param ds_param_crop_size: Dataset transform - image crop size
        :type ds_param_crop_size: int
        :param ds_param_crop_padding: Dataset transform - image crop padding
        :type ds_param_crop_padding: int
        :param ds_param_means: Dataset transform - normalization means
        :type ds_param_means: List[float]
        :param ds_param_stds: Dataset transform - normalization standard deviations
        :type ds_param_stds: List[float]
        :param lrf_device: Used device type for lr finding
        :type lrf_device: EDeviceType | type(None)
        :param lrf_criterion: Criterion (loss) function for lr finding
        :type lrf_criterion: ECriterion
        :param lrf_optimizer: Optimizer function for lr finding
        :type lrf_optimizer: EOptimizer
        :param lrf_start_lr: Starting lr for lr finding
        :type lrf_start_lr: float
        :param lrf_end_lr: Ending lr for lr finding
        :type lrf_end_lr: float
        :param lrf_num_iter: Number of max iterations for lr finding
        :type lrf_num_iter: int
        :param param_preset: Preset VGG layers configuration
        :type param_preset: dict[str, List[int | str]]
        :param param_ft_in_channels: Input conv. channels
        :type param_ft_in_channels: int
        :param param_ft_kernel_size: Conv. layers kernel size
        :type param_ft_kernel_size: int
        :param param_ft_pool_kernel_size: Pooling layers kernel size
        :type param_ft_pool_kernel_size: int
        :param param_ft_padding: Conv. layers padding
        :type param_ft_padding: int
        :param param_ft_is_batchnorm_used: Is used batch normalization
        :type param_ft_is_batchnorm_used: bool
        :param param_avg_pool_size: Average pooling size between features and classifier modules
        :type param_avg_pool_size: int
        :param param_clf_dims: Input/output dimensions for each dense layer
        :type param_clf_dims: List[Tuple[int, int]]
        :param param_clf_dropout: Dense layer dropout
        :type param_clf_dropout: float
        :param hparam_device: Used device type
        :type hparam_device: EDeviceType | type(None)
        :param hparam_batch_size: Size of the data batches
        :type hparam_batch_size: int
        :param hparam_criterion: Criterion (loss) function
        :type hparam_criterion: ECriterion
        :param hparam_optimizer: Optimizer function
        :type hparam_optimizer: EOptimizer
        :param hparam_learning_rate: Model learning rate
        :type hparam_learning_rate: float
        :param hparam_epochs: Number of training epochs
        :type hparam_epochs: int
        """
        self.name = name
        self.description = description
        self.utilized_dataset = utilized_dataset
        self.ds_param_random_rotation = ds_param_random_rotation
        self.ds_param_random_horizontal_flip = ds_param_random_horizontal_flip
        self.ds_param_crop_size = ds_param_crop_size
        self.ds_param_crop_padding = ds_param_crop_padding
        self.ds_param_means = ds_param_means
        self.ds_param_stds = ds_param_stds
        self.lrf_device = lrf_device
        self.lrf_criterion = lrf_criterion
        self.lrf_optimizer = lrf_optimizer
        self.lrf_start_lr = lrf_start_lr
        self.lrf_end_lr = lrf_end_lr
        self.lrf_num_iter = lrf_num_iter
        self.param_preset = param_preset
        self.param_ft_in_channels = param_ft_in_channels
        self.param_ft_kernel_size = param_ft_kernel_size
        self.param_ft_pool_kernel_size = param_ft_pool_kernel_size
        self.param_ft_padding = param_ft_padding
        self.param_ft_is_batchnorm_used = param_ft_is_batchnorm_used
        self.param_avg_pool_size = param_avg_pool_size
        self.param_clf_dims = param_clf_dims
        self.param_clf_dropout = param_clf_dropout
        self.hparam_device = hparam_device
        self.hparam_batch_size = hparam_batch_size
        self.hparam_criterion = hparam_criterion
        self.hparam_optimizer = hparam_optimizer
        self.hparam_learning_rate = hparam_learning_rate
        self.hparam_epochs = hparam_epochs

    @staticmethod
    def from_json():
        """ Returns config object based on config.json file """
        config_dict = json.load(open("config.json"))
        nn_meta_dict = config_dict['vgg']
        nn_ds_param_dict = nn_meta_dict['ds_param']
        nn_lrf_dict = nn_meta_dict['lrf']
        nn_param_preset_dict = nn_meta_dict['param']['preset']
        nn_param_ft_dict = nn_meta_dict['param']['ft']
        nn_param_avg_pool_size = nn_meta_dict['param']['avg_pool_size']
        nn_param_clf_dict = nn_meta_dict['param']['clf']
        nn_hparam_dict = nn_meta_dict['hparam']
        return VGGConfig(nn_meta_dict['name'],
                         nn_meta_dict['description'],
                         EDataset[nn_meta_dict['utilized_dataset']],
                         nn_ds_param_dict['random_rotation'],
                         nn_ds_param_dict['random_horizontal_flip'],
                         nn_ds_param_dict['crop_size'],
                         nn_ds_param_dict['crop_padding'],
                         nn_ds_param_dict['means'],
                         nn_ds_param_dict['stds'],
                         EDeviceType[nn_lrf_dict['device']] if nn_lrf_dict['device'] is not None else None,
                         ECriterion[nn_lrf_dict['criterion']],
                         EOptimizer[nn_lrf_dict['optimizer']],
                         nn_lrf_dict['start_lr'],
                         nn_lrf_dict['end_lr'],
                         nn_lrf_dict['num_iter'],
                         nn_param_preset_dict,
                         nn_param_ft_dict['in_channels'],
                         nn_param_ft_dict['kernel_size'],
                         nn_param_ft_dict['pool_kernel_size'],
                         nn_param_ft_dict['padding'],
                         nn_param_ft_dict['is_batchnorm_used'],
                         nn_param_avg_pool_size,
                         nn_param_clf_dict['dims'],
                         nn_param_clf_dict['dropout'],
                         EDeviceType[nn_hparam_dict['device']] if nn_hparam_dict['device'] is not None else None,
                         nn_hparam_dict['batch_size'],
                         ECriterion[nn_hparam_dict['criterion']],
                         EOptimizer[nn_hparam_dict['optimizer']],
                         nn_hparam_dict['learning_rate'],
                         nn_hparam_dict['epochs'])

    def to_dict(self):
        """ Returns dictionary of config parameters """
        config_dict = dict(map(lambda x: (x, getattr(self, x)),
                               filter(lambda x: not x.startswith('_') and not callable(getattr(self, x)),
                                      dir(self))))
        return config_dict

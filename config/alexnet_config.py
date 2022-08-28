import json

from models.e_dataset import EDataset
from models.e_device_type import EDeviceType
from models.e_criterion import ECriterion
from models.e_optimizer import EOptimizer


class AlexNetConfig:
    def __init__(self, name, description, utilized_dataset,
                 param_ft_in_out_channels, param_ft_kernel_size, param_ft_pool_kernel_size,
                 param_ft_stride, param_ft_padding, param_clf_dims, param_clf_dropout,
                 hparam_device, hparam_batch_size, hparam_criterion,
                 hparam_optimizer, hparam_learning_rate, hparam_epochs):
        """
        AlexNet NN configuration
        :param name: Name of the model
        :type name: str
        :param description: Model description
        :type description: str
        :param utilized_dataset: Dataset for model training and testing
        :type utilized_dataset: EDataset
        :param param_ft_in_out_channels: Input/output channels for each conv. layer
        :type param_ft_in_out_channels: List[Tuple[int, int]]
        :param param_ft_kernel_size: Conv. layers kernel size
        :type param_ft_kernel_size: int
        :param param_ft_pool_kernel_size: Pooling layers kernel size
        :type param_ft_pool_kernel_size: int
        :param param_ft_stride: Conv. layers stride
        :type param_ft_stride: int
        :param param_ft_padding: Conv. layers padding
        :type param_ft_padding: int
        :param param_clf_dims: Input/output dimensions for each dense layer
        :type param_clf_dims: List[Tuple[int, int]]
        :param param_clf_dropout: Dense layers dropout
        :type param_clf_dropout: float
        :param hparam_device: Used device type
        :type hparam_device: EDeviceType
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
        self.param_ft_in_out_channels = param_ft_in_out_channels
        self.param_ft_kernel_size = param_ft_kernel_size
        self.param_ft_pool_kernel_size = param_ft_pool_kernel_size
        self.param_ft_stride = param_ft_stride
        self.param_ft_padding = param_ft_padding
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
        nn_meta_dict = config_dict['alexnet']
        nn_param_ft_dict = nn_meta_dict['param']['ft']
        nn_param_clf_dict = nn_meta_dict['param']['clf']
        nn_hparam_dict = nn_meta_dict['hparam']
        return AlexNetConfig(nn_meta_dict['name'],
                             nn_meta_dict['description'],
                             EDataset[nn_meta_dict['utilized_dataset']],
                             nn_param_ft_dict['in_out_channels'],
                             nn_param_ft_dict['kernel_size'],
                             nn_param_ft_dict['pool_kernel_size'],
                             nn_param_ft_dict['stride'],
                             nn_param_ft_dict['padding'],
                             nn_param_clf_dict['dims'],
                             nn_param_clf_dict['dropout'],
                             EDeviceType[nn_hparam_dict['device']],
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

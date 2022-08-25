import json

from models.e_dataset import EDataset
from models.e_device_type import EDeviceType
from models.e_criterion import ECriterion
from models.e_optimizer import EOptimizer


class DenseNNConfig:
    def __init__(self, name, description, utilized_dataset,
                 param_input_dim, param_hidden_dims, param_output_dim,
                 hparam_device, hparam_batch_size, hparam_criterion,
                 hparam_optimizer, hparam_learning_rate, hparam_epochs):
        """
        Dense NN configuration
        :param name: Name of the model
        :type name: str
        :param description: Model description
        :type description: str
        :param utilized_dataset: Dataset for model training and testing
        :type utilized_dataset: EDataset
        :param param_input_dim: Input size
        :type param_input_dim: int
        :param param_hidden_dims: Hidden sizes
        :type param_hidden_dims: List[int]
        :param param_output_dim: Output size
        :type param_output_dim: int
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
        self.param_input_dim = param_input_dim
        self.param_hidden_dims = param_hidden_dims
        self.param_output_dim = param_output_dim
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
        nn_meta_dict = config_dict['mlp']
        nn_param_dict = nn_meta_dict['param']
        nn_hparam_dict = nn_meta_dict['hparam']
        return DenseNNConfig(nn_meta_dict['name'],
                             nn_meta_dict['description'],
                             EDataset[nn_meta_dict['utilized_dataset']],
                             nn_param_dict['input_dim'],
                             nn_param_dict['hidden_dims'],
                             nn_param_dict['output_dim'],
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

from models.dict_criterion import criterions_dict
from models.dict_dataset import datasets_dict
from models.dict_lr_scheduler import lr_schedulers_dict
from models.dict_optimizer import optimizer_dict
from models.dict_pretrained_model import pretrained_models_dict


def get_criterion(key):
    return criterions_dict[key]


def get_dataset(key):
    return datasets_dict[key]


def get_lr_scheduler(key):
    return lr_schedulers_dict[key]


def get_optimizer(key):
    return optimizer_dict[key]


def get_pretrained_model(key):
    return pretrained_models_dict[key]

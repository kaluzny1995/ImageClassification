import torch.nn as nn


criterions_dict = dict(
    ce=nn.CrossEntropyLoss,
    bce=nn.BCELoss,
    bcel=nn.BCEWithLogitsLoss
)

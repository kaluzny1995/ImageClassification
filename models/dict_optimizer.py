import torch.optim as optim


optimizer_dict = dict(
    adam=optim.Adam,
    adamax=optim.Adamax,
    adagrad=optim.Adagrad
)

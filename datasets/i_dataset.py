from abc import ABCMeta, abstractmethod


class IDataset(metaclass=ABCMeta):
    @abstractmethod
    def get_datasets(self):
        """ Returns the training, validation and testing datasets. """
        pass

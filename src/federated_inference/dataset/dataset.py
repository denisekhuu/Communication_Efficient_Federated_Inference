from abc import abstractmethod
from collections.abc import Iterable

# TODO refactor testloader and train_loader
class Dataset(): 
    
    def __init__(self, config, transform_methods : Iterable = []):
        self.config = config
        self.transform_methods = transform_methods
        self.train_dataset = self.load_train_data()
        self.test_dataset  = self.load_test_data()
    
    @abstractmethod
    def load_train_data(self):
        """
        Loads & returns the training dataloader and dataset.

        :return: torchvision.datasets.Dataset
        """
        raise NotImplementedError("load_train_dataloader() isn't implemented")
        
    @abstractmethod
    def load_test_data(self):
        """
        Loads & returns the test dataloader and dataset. 

        :return: torchvision.datasets.Dataset
        """
        raise NotImplementedError("load_test_dataloader() isn't implemented")
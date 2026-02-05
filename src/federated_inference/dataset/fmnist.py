import torch, torchvision
from .dataset import Dataset
from collections.abc import Iterable

class FMNISTDataset(Dataset): 
    
    def __init__(self, config, transform_methods : Iterable = []):
        super(FMNISTDataset, self).__init__(config, transform_methods)
        self.name = "Fashion MNIST"
        self.path = self.config.FMNIST_DATASET_PATH
        self.labels = self.config.FMNIST_LABELS
        
        
    def load_train_data(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,)), 
            *self.transform_methods
        ])
        
        train_dataset = torchvision.datasets.FashionMNIST(
            self.config.FMNIST_DATASET_PATH, 
            train=True, download=True,
            transform=transform)
        
        
        print("FashionMnist training data loaded.")
        return train_dataset
    
    def load_test_data(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,)),
                *self.transform_methods
        ])
        
        test_dataset = torchvision.datasets.FashionMNIST(
            self.config.FMNIST_DATASET_PATH, 
            train=False, download=True,
            transform=transform)
        
        print("FashionMnist training data loaded.")
        return test_dataset

        
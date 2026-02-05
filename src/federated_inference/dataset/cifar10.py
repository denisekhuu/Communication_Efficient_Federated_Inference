import torch, torchvision
from .dataset import Dataset
from collections.abc import Iterable

class CIFAR10Dataset(Dataset): 
    
    def __init__(self, config, transform_methods : Iterable = []):
        super(CIFAR10Dataset, self).__init__(config, transform_methods)
        self.name = "CIFAR 10"
        self.path = self.config.CIFAR10_DATASET_PATH
        self.labels = self.config.CIFAR10_LABELS
        
    def load_train_data(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.transforms.ToTensor(), 
            torchvision.transforms.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            *self.transform_methods])
        
        train_dataset = torchvision.datasets.CIFAR10(
            self.config.CIFAR10_DATASET_PATH, 
            train=True, download=True,
            transform=transform)
        
        print("CIFAR10 training data loaded.")
        return train_dataset
    
    def load_test_data(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.transforms.ToTensor(),
            torchvision.transforms.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            *self.transform_methods])
        
        test_dataset = torchvision.datasets.CIFAR10(
            self.config.CIFAR10_DATASET_PATH, 
            train=False, download=True,
            transform=transform)
        
        print("CIFAR10 test data loaded.")
        return test_dataset

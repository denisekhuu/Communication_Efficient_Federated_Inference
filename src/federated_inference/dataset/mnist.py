import torch, torchvision
from .dataset import Dataset
from collections.abc import Iterable

class MNISTDataset(Dataset): 
    
    def __init__(self, config, transform_methods : Iterable = []):
        super(MNISTDataset, self).__init__(config, transform_methods)
        self.name = "MNIST"
        self.path = config.MNIST_DATASET_PATH
        
    def load_train_data(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.transforms.ToTensor(), 
            *self.transform_methods])
        
        train_dataset = torchvision.datasets.MNIST(
            self.config.MNIST_DATASET_PATH, 
            train=True, download=True,
            transform=transform)
        
        print("MNIST training data loaded.")
        return train_dataset
    
    def load_test_data(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.transforms.ToTensor(), 
            *self.transform_methods])
        
        test_dataset = torchvision.datasets.MNIST(
            self.config.MNIST_DATASET_PATH, 
            train=False, download=True,
            transform=transform)
        
        print("MNIST test data loaded.")
        return test_dataset
    

import torch, torchvision
from torch.utils.data import Dataset
from collections.abc import Iterable

class ClientDataset(): 
    def __init__(self, config, datasets: tuple[Dataset, Dataset], transform_methods : Iterable = []):
        self.config = config
        self.name = self.config.DATASET_NAME
        self.path = self.config.DATASET_PATH 
        self.labels = self.config.LABELS
        self.transform_methods = transform_methods
        self.train_dataset = datasets[0]
        self.test_dataset = datasets[1]

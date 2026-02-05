from federated_inference.client.sensor_view import SensorView
from collections.abc import Iterable
from torch.utils.data import Dataset
from torch import tensor

class SensorWrapper(Dataset):
  def __init__(self, dataset: Dataset, idx: int , transform = None):
    self.dataset = dataset
    self.targets = dataset.targets
    self.transform = transform
    self.idx = idx

  def _select_sensorview(self, sensor_views: Iterable[SensorView]):
    return sensor_views[self.idx]

  def __getitem__(self, index: int) -> SensorView:
    data, target = self.dataset[index]
    data = self._select_sensorview(data).evidence
    if self.transform: 
      data = self.transform(data)
    return data, target

  def __len__(self):
    return len(self.dataset)

class DatasetWrapper(Dataset):
  def __init__(self, dataset: Dataset, transform = None):
    self.dataset = dataset
    self.targets = dataset.targets
    self.transform = transform

  def __getitem__(self, index: int) -> SensorView:
    data, target = self.dataset[index]
    if self.transform: 
      data = self.transform(data)
    return data, target

  def __len__(self):
    return len(self.dataset)
    
if __name__ == "__main__": 
    import torchvision
    import os 
    import torch
    from federated_inference.dataset import MNISTDataset 
    import random 

    configs = Configuration()
    data = MNISTDataset(configs)

    mask_size = [14,14]
    dimensions =  [1,2] 
    stride = 14
    op_transform = StridePartitionTransform('full', mask_size, dimensions, stride)
    transform = torchvision.transforms.Compose([torchvision.transforms.transforms.ToTensor(), op_transform])
    MNIST_DATASET_PATH = os.path.join('./data/mnist')
    train_dataset = torchvision.datasets.MNIST(
        MNIST_DATASET_PATH, 
        train=True, download=True,
        transform=transform)

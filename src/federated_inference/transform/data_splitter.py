import torch
from torch.utils.data import DataLoader, Subset, Dataset
from collections.abc import Iterable
from typing import List
from collections import defaultdict
from itertools import permutations
import math
import matplotlib.pyplot as plt
from itertools import chain
import collections
import logging
import random 

class DataSplitter(): 

  @staticmethod
  def group_dataset(dataset: torch.utils.data.Dataset):

      label_to_indices = defaultdict(list)
      for idx in range(len(dataset)):
          _, label = dataset[idx]
          label_to_indices[label].append(idx)
        
      return label_to_indices



  @staticmethod
  def split_dataset_by_distribution(dataset: torch.utils.data.Dataset, n:int, p: List[float], seed=None):
      """
      Splits a dataset to n subsets which follow a distribution p      
      Args:
          dataset: a PyTorch dataset
          n: number of clients
          p: prefered probability distribution over labels
          seed: optional, for reproducibility
      
      Returns:
          Subset of the dataset with the desired label distribution
      """
      if sum(p) != 1:
        raise ValueError("ValueError: p needs to sum to 1") 

      if seed is not None:
          random.seed(seed)

      label_to_indices = DataSplitter.group_dataset(dataset)
      client_set_size = math.floor(len(dataset)/n)
      keys = list(label_to_indices.keys())
      clients = []

      for _n in range(n):
        random_labels = random.sample(keys, len(keys))
        data_selection = []
        for _p in p:  
          for key in random_labels: 
            if len(label_to_indices[key]) >= client_set_size*_p: 
              selected = random.sample(list(label_to_indices[key]), round(client_set_size*_p))
              data_selection += selected 
              label_to_indices[key] = set(label_to_indices[key]) - set(selected)
              random_labels.remove(key)
              break

        if len(data_selection) < client_set_size:
          logging.info(f"Warning: subset {_n} does not follow the distribution strictly")
          rest = list(chain(*label_to_indices.values()))[:round((client_set_size)-len(data_selection))]
          data_selection += rest
          for key in random_labels: 
            label_to_indices[key] = set(label_to_indices[key]) - set(rest)
        clients.append(torch.utils.data.Subset(dataset, data_selection))
      return clients

  @staticmethod
  def divide_data_equally(dataset: torch.utils.data.Dataset, n: int) -> List[torch.utils.data.Subset]:
      """
      Divides the dataset into n different subsets such that all images are distributed
      The data 
      return torch.utils.data.Subset[]
      """
      indices = [[] for i in range(n)]
      for i in range(len(dataset)):
          indices[i % n].append(i)
      trainsets = [torch.utils.data.Subset(dataset, idx) for idx in indices]
      return trainsets

  @staticmethod
  def create_dist_loaders(distributed_datasets: List[torch.utils.data.Subset], batch_size: int | Iterable) -> List[torch.utils.data.DataLoader]:
      """
      Turns a set torch.utils.data.Subset into DataLoaders
      return torch.utils.data.DataLoader[]
      """
      dataloaders = []
      if isinstance(batch_size, int):
        dataloaders = [
            torch.utils.data.DataLoader(_set, batch_size=batch_size,shuffle=True, num_workers=2)
            for _set in distributed_datasets
        ]
      elif isinstance(batch_size, list):
        if len(batch_size) != distributed_datasets:
          assert Exception("batch_size and distributed_datasets needs to be of equal size")
        dataloaders = [
            torch.utils.data.DataLoader(_set, batch_size=batch_size, shuffle=True, num_workers=2)
            for _set in distributed_datasets
        ]
      else: 
        assert Exception("batch_size needs to be an int or a list of integers")
      return dataloaders
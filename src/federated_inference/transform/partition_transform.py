from typing import List
import random
from itertools import product
from collections.abc import Iterable
import torch
from federated_inference.client.sensor_view import SensorView

class StridePartitionTransform:

    def __init__(self, config):
        """
            method_name: str ('fixed_number', 'drop_probability', 'full')
            if method 'fixed_number' : 
                n_positions: number of sensor position to keep
            if method 'drop_probability':
                drop_p: the drop probability of a position

        """
        self.method_name = config.METHOD_NAME
        self.mask_size = config.MASK_SIZE
        self.dimensions = config.DIMENSIONS
        self.stride = config.STRIDE
        self.n_position = config.N_POSITION
        self.drop_p = config.DROP_P
        self.tensor_size = config.TENSOR_SIZE
        self.positions = self._random_stride_positions(config.TENSOR_SIZE)
        self.coverage = self._calculate_coverage_from_config(
            tensor_size=self.tensor_size,
            mask_size=self.mask_size,
            dimensions=self.dimensions,
            positions=self.positions)
        self.coverage_percent = self.compute_coverage_percent()
        self.coverage_sensors = torch.prod(torch.tensor(self.mask_size), 0)/torch.prod(torch.tensor(self.tensor_size), 0)


    def __len__(self) -> int:
        return len(self.positions)

    def _calculate_coverage_from_config(self, tensor_size, mask_size, dimensions, positions):
        coverage = torch.zeros(tensor_size)

        for pos in positions:
            index_slices = [slice(None)] * len(tensor_size)
            for dim_idx, jump in enumerate(pos):
                dim = dimensions[dim_idx]
                size = mask_size[dim_idx]
                index_slices[dim] = slice(jump, jump + size)
            
            # Increment the coverage map in the region of the mask
            coverage[tuple(index_slices)] += 1

        return coverage

    def compute_coverage_percent(self) -> float:
        total_elements = self.coverage.numel()
        covered_elements = (self.coverage > 0).sum().item()
        return 100.0 * covered_elements / total_elements

    def _random_stride_positions(self, size, seed=None) -> list: 
        if seed is not None:
            random.seed(seed)
        if isinstance(self.stride, Iterable):
            if len(self.stride) != len(self.dimensions) and len(self.mask_size) != len(self.dimensions) :
                assert Exception("Size Error: Dimension missmatch. stride, n_grids and dimensions need to be of equal size. ")
        if self.method_name == 'fixed_number': 
            if not self.n_position: 
                assert Exception("Value Error: Number of n_positions was not set")
            positons = list(product(*[range(0, size[dim]-self.mask_size[idx]+1, self.stride if isinstance(self.stride, int) else self.stride[idx]) for idx, dim in enumerate(self.dimensions)]))
            return random.sample(positons, self.n_position)
        elif self.method_name == 'drop_probability': 
            if not self.drop_p:
                assert Exception("Value Error: drop_p was not set")
            positons = list(product(*[range(0, size[dim]-self.mask_size[idx]+1, self.stride if isinstance(self.stride, int) else self.stride[idx]) for idx, dim in enumerate(self.dimensions)]))
            return [positons[idx] for idx, p in enumerate(random.choices([0, 1], [1 - self.drop_p, self.drop_p], k=len(positons))) if p]
        elif self.method_name == 'full':
            return list(product(*[range(0, size[dim]-self.mask_size[idx]+1, self.stride if isinstance(self.stride, int) else self.stride[idx]) for idx, dim in enumerate(self.dimensions)]))       
        else:
            assert Exception("You need to specify a method and stride needs to be an Integer or a list of Integers")

    def _slice_stride_partition(self, tensor) -> List[dict]:
        """
            Divides an image into subimages 

            ## Input
            tensor: torch.tensor (image)

            ## Output
            list of subimages: List[torch.tensor]

        """
        tensors = []
        for jumps in self.positions:
            index_slices = [slice(None)] * tensor.ndim
            for jump_index, jump in enumerate(jumps):
                dim = self.dimensions[jump_index]
                size = self.mask_size[jump_index]
                index_slices[dim] = slice(jump, jump + size)
            sub_tensor = tensor[tuple(index_slices)]
            tensors.append(SensorView(sub_tensor, index_slices, tensor.size()))
        return tensors

    def __call__(self, tensor) -> List[dict]:
        return self._slice_stride_partition(tensor)

if __name__ == "__main__":

    import torchvision
    import os 
    import torch
    from federated_inference.configs.config import Configuration
    from federated_inference.dataset import MNISTDataset,  Dataset
    from federated_inference.transform.utils import show_sensorviews
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

    random_index = random.randint(0, len(train_dataset) - 1)
    image, label = train_dataset[random_index]
    show_sensorviews(image)

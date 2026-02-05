import torch
import random
from typing import List
from federated_inference.client.sensor_view import SensorView


class OneDStridePartitionTransform:
    def __init__(self, config):
        """
        Config should have:
        - METHOD_NAME: 'fixed_number', 'drop_probability', 'full'
        - MASK_SIZE: int
        - STRIDE: int
        - N_POSITION: int (if fixed_number)
        - DROP_P: float (if drop_probability)
        - TENSOR_SIZE: int
        """
        self.method_name = config.METHOD_NAME
        self.mask_size = config.MASK_SIZE
        self.stride = config.STRIDE
        self.n_position = getattr(config, "N_POSITION", None)
        self.drop_p = getattr(config, "DROP_P", None)
        self.tensor_size = config.TENSOR_SIZE

        self.positions = self._compute_positions()
    
    def _compute_positions(self):
        max_start = self.tensor_size - self.mask_size + 1
        if max_start <= 0:
            raise ValueError("Mask size too large for the given tensor size.")

        all_positions = list(range(0, max_start, self.stride))

        if self.method_name == 'fixed_number':
            if not self.n_position:
                raise ValueError("n_position must be set for method 'fixed_number'")
            return random.sample(all_positions, min(self.n_position, len(all_positions)))

        elif self.method_name == 'drop_probability':
            if self.drop_p is None:
                raise ValueError("drop_p must be set for method 'drop_probability'")
            return [pos for pos in all_positions if random.random() > self.drop_p]

        elif self.method_name == 'full':
            return all_positions

        else:
            raise ValueError(f"Unknown method_name: {self.method_name}")

    def _slice_tensor(self, tensor: torch.Tensor) -> List[SensorView]:
        """
        Slices a 1D tensor into views.
        """
        if tensor.ndim != 1:
            raise ValueError("Expected 1D tensor")

        views = []
        for pos in self.positions:
            end = pos + self.mask_size
            if end <= tensor.shape[0]:
                slice_tensor = tensor[pos:end]
                index_slice = slice(pos, end)
                views.append(SensorView(slice_tensor, index_slice, tensor.shape))

        return views

    def __call__(self, tensor: torch.Tensor) -> List[SensorView]:
        return self._slice_tensor(tensor)

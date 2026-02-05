import torch
from torchvision import datasets, transforms
from PIL import Image
import io
from torch.utils.data import Dataset
from torch import Tensor
import logging


logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG, INFO, WARNING, etc.
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CommunicationCost(): 
    def __init__(self, total_tensor_bytes, total_png_bytes, total_target_in_bytes, n_dataset):
        self.total_tensor_bytes = total_tensor_bytes
        self.total_png_bytes = total_png_bytes
        self.total_target_in_bytes = total_target_in_bytes
        self.n_dataset = n_dataset

    def set_cost_reason(self, reason: str, log: bool=False):
        if log:
            logging.info(f"communication_reason: {reason}")
        self.reason = reason

    def __dict__(self):
        return {
            'reason': self.reason,
            'n_dataset': self.n_dataset,
            'total_tensor_bytes' : self.total_tensor_bytes, 
            'total_png_bytes': self.total_png_bytes,
            'total_target_bytes' : self.total_target_in_bytes,
        }

class CostCalculator():

    @staticmethod
    def calculate_communication_cost_by(dataset, use_png:bool = False, log:bool = False):
        total_png_bytes = 0
        total_tensor_bytes = 0
        total_target_in_bytes = 0

        if isinstance(dataset, Dataset):
            for i in range(len(dataset)):
                tensor, target = dataset[i] 
                total_tensor_bytes += tensor.element_size() * tensor.nelement()
                if use_png:
                    image = transforms.ToPILImage()(tensor)
                    buffer = io.BytesIO()
                    image.save(buffer, format='PNG')
                    png_size = buffer.tell()
                    total_png_bytes += png_size

            tensor_size_MB = total_tensor_bytes / (1024 ** 2)
            png_size_MB = total_png_bytes / (1024 ** 2)

            targets = dataset.targets if isinstance(dataset.targets, torch.Tensor) else torch.tensor(dataset.targets)
            total_target_in_bytes = targets.element_size() * targets.nelement()
            target_size_MB = total_target_in_bytes / (1024 ** 2)

            if log:
                logging.info(f"Total raw tensor size: {tensor_size_MB:.2f} MB")
                logging.info(f"Estimated total PNG size: {png_size_MB:.2f} MB")
                logging.info(f"Total raw target size: {target_size_MB:.2f} MB")
            return CommunicationCost(total_tensor_bytes, total_png_bytes, total_target_in_bytes, len(dataset))

        if isinstance(dataset, tuple) or isinstance(dataset, Tensor):
            if isinstance(dataset, tuple):
                tensor, target = dataset
                total_target_in_bytes = target.element_size()
                if log:
                    logging.info(f"Total raw target size: {total_target_in_bytes:.2f} B")
                return CommunicationCost(total_tensor_bytes, total_png_bytes, total_target_in_bytes, 1)
            elif isinstance(dataset, Tensor):
                tensor = dataset
            total_tensor_bytes = tensor.element_size() * tensor.nelement()
            image = transforms.ToPILImage()(tensor)
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            png_size = buffer.tell()
            total_png_bytes = png_size
            if log:
                logging.info(f"Total raw tensor size (bytes B): {total_tensor_bytes:.2f} B")
                logging.info(f"Estimated total PNG size (bytes B): {total_png_bytes:.2f} B")
            return CommunicationCost(total_tensor_bytes, total_png_bytes, total_target_in_bytes, 1)

        if isinstance(dataset, list):
            total_tensor_bytes = len(dataset)*torch.tensor(dataset).element_size()
            if log:
                logging.info(f"Total raw tensor size (bytes B): {total_tensor_bytes:.2f} B")
            return CommunicationCost(total_tensor_bytes, total_png_bytes, total_target_in_bytes, len(dataset))
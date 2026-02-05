import math
import torch
import matplotlib.pyplot as plt
from typing import List, Union, Tuple
from federated_inference.client.sensor_view import SensorView
from torch.utils.data import Subset, Dataset

def show_sensorviews(sensor_views: List[SensorView]):
    if len(sensor_views) > 3:
        cols, rows = math.floor(len(sensor_views)/2), math.ceil(len(sensor_views)/2)
    else: 
        cols, rows = 1, len(sensor_views)
    figure = plt.figure(figsize=(8, 8))
    for i, view in enumerate(sensor_views): 
        figure.add_subplot(cols, rows, i+1)
        plt.imshow(view.evidence.squeeze(), cmap="gray")
    plt.show()
    plt.close()

def show_img(image: Union[torch.Tensor, Tuple[torch.Tensor, int], SensorView], cmap: str ="gray"):
    if isinstance(image, SensorView): 
        plt.imshow(image.evidence.squeeze(), cmap=cmap)
    elif isinstance(image, torch.Tensor):
        plt.imshow(image.squeeze(), cmap=cmap)
    else: 
        plt.imshow(image[0].squeeze(), cmap=cmap)
    plt.axis('off')  # Optional: remove axes
    plt.show()
    plt.close()

def show_label_distribution(dataset: Dataset|Subset): 
    labels = [label for _, label in dataset]
    data = collections.Counter(labels)

    # Extract keys and values
    keys = list(data.keys())
    values = list(data.values())
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(keys, values, color='skyblue')
    plt.xlabel('Labels')
    plt.ylabel('Amount')
    plt.title('Label Count')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    plt.close()
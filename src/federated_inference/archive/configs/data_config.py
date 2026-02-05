import os
from dotenv import load_dotenv
from federated_inference.common.environment import DataSetEnum

class DataConfiguration():
    
    #MNIST_FASHION_DATASET Configurations
    FMNIST_NAME = 'FMNIST'
    FMNIST_DATASET_PATH = os.path.join('./data/fmnist')
    FMNIST_LABELS = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',  'Bag', 'Ankle Boot']
    
    #MNIST_DATASET Configurations
    MNIST_NAME = 'MNIST'
    MNIST_DATASET_PATH = os.path.join('./data/mnist')
    
    #CIFAR_DATASET Configurations
    CIFAR10_NAME = 'CIFAR10'
    CIFAR10_DATASET_PATH = os.path.join('./data/cifar10')
    CIFAR10_LABELS = ['Plane', 'Car', 'Bird', 'Cat','Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    def __init__(self, dataset_name : str = None):
        load_dotenv(override=True)
        self.DATASET_NAME = dataset_name if dataset_name else os.getenv('DATASET_NAME', 'MNIST')
        if self.DATASET_NAME == DataSetEnum.MNIST.value: 
            self.INSTANCE_SIZE = (1, 28, 28)
            self.DATASET = DataSetEnum.MNIST
            self.DATASET_PATH = os.path.join(os.getenv('DATASET_PATH', self.MNIST_DATASET_PATH))
            self.LABELS = range(10)
        if self.DATASET_NAME == DataSetEnum.FMNIST.value: 
            self.INSTANCE_SIZE = (1, 28, 28)
            self.DATASET = DataSetEnum.FMNIST
            self.DATASET_PATH = os.path.join(os.getenv('DATASET_PATH', self.FMNIST_DATASET_PATH))
            self.LABELS = self.FMNIST_LABELS
        if self.DATASET_NAME == DataSetEnum.CIFAR10.value:
            self.INSTANCE_SIZE = (1, 32, 32)
            self.DATASET = DataSetEnum.CIFAR10
            self.DATASET_PATH = os.path.join(os.getenv('DATASET_PATH', self.CIFAR10_DATASET_PATH))
            self.LABELS =  self.CIFAR10_LABELS

    def __dict__(self):
        return {
            "dataset_name": self.DATASET_NAME, 
            "labels": list(self.LABELS),
            "size": self.INSTANCE_SIZE
        }
from federated_inference.common.environment import DataDistributionType, TransformType, DataMode, DataSetEnum
from torch.utils.data import Dataset as TorchDataset
from federated_inference.transform import DataSplitter, SensorWrapper, DatasetWrapper
from federated_inference.dataset import ClientDataset

from federated_inference.simulations.naive.configs.data_config import DataConfiguration
from federated_inference.simulations.naive.configs.transform_config import DataTransformConfiguration
from federated_inference.dataset import MNISTDataset, FMNISTDataset, CIFAR10Dataset 


class Simulation():

    def __init__(self):
        self.data_config = None
        self.transform_config = None
        self.data_mode = None
        self.transform_type = None

    def load_data(self, data_config: DataConfiguration):
        dataset =  TorchDataset()
        if data_config.DATASET == DataSetEnum.MNIST: 
            dataset = MNISTDataset(data_config)
        elif data_config.DATASET == DataSetEnum.FMNIST: 
            dataset = FMNISTDataset(data_config)
        elif data_config.DATASET == DataSetEnum.CIFAR10: 
            dataset = CIFAR10Dataset(data_config)
        return dataset

    def transform_data(self, dataset: TorchDataset, data_distribution_type: DataDistributionType | None = None, data_mode: DataMode | None = None, transform_type: TransformType | None = None, transform_config: DataTransformConfiguration| None = None): 
        transformation = None
        if data_distribution_type and data_mode and data_mode == DataMode.HORIZONTAL and data_distribution_type == DataDistributionType.DIVIDE_EQUALLY:
            dataset = tuple(zip(DataSplitter.divide_data_equally(dataset.train_dataset, env_config.N_CLIENTS), DataSplitter.divide_data_equally(dataset.test_dataset, env_config.N_CLIENTS)))
            dataset = [ClientDataset(self.data_config, d) for d in dataset]
        if data_mode and data_mode == DataMode.VERTICAL and transform_type and transform_config:
            if transform_type == TransformType.FULL_STRIDE_PARTITION:
                from federated_inference.transform import StridePartitionTransform
                transformation = StridePartitionTransform(transform_config)
            dataset = [ClientDataset(self.data_config, (
                SensorWrapper(DatasetWrapper(dataset.train_dataset, transformation), i), 
                SensorWrapper(DatasetWrapper(dataset.test_dataset, transformation), i))) for i in range(len(transformation))]
        return dataset, transformation


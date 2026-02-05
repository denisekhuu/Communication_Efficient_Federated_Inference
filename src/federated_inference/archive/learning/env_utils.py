from federated_inference.configs.data_config import DataConfiguration
from federated_inference.configs.torch_config import TorchConfiguration
from federated_inference.configs.env_config import EnvConfiguration
from federated_inference.configs.model_config import ModelConfiguration
from federated_inference.configs.member_config import MemberConfiguration
from federated_inference.dataset import MNISTDataset, FMNISTDataset, CIFAR10Dataset, Dataset, ClientDataset
from federated_inference.transform import DataSplitter, SensorWrapper, DatasetWrapper
from federated_inference.learning.FL.utils import ModelAggregator, ClientSelector
from federated_inference.learning import CentralMNISTModel, SplitMNISTClientModel, SplitMNISTServerModel, FederatedMNISTModel, FederatedMNISTClientModel, FederatedMNISTServerModel
import torch.optim as optim
from .environment import LearningMethod, DataSetEnum, Member, DataDistributionType, TransformType
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset
from collections.abc import Iterable

class EnvUtils():

    @staticmethod
    def load_data(data_config: DataConfiguration, env_config: EnvConfiguration | None = None) -> Dataset | Iterable[Dataset]:
        dataset =  TorchDataset()
        if data_config.DATASET == DataSetEnum.MNIST: 
            dataset = MNISTDataset(data_config)
        elif data_config.DATASET == DataSetEnum.FMNIST: 
            dataset = FMNISTDataset(data_config)
        elif data_config.DATASET == DataSetEnum.CIFAR10: 
            dataset = CIFAR10Dataset(data_config)
        if env_config and env_config.LEARNING_METHOD == LearningMethod.HFL and hasattr(env_config, 'data_distribution_type') and env_config.data_distribution_type == DataDistributionType.DIVIDE_EQUALLY:
            dataset = tuple(zip(DataSplitter.divide_data_equally(dataset.train_dataset, env_config.N_CLIENTS), DataSplitter.divide_data_equally(dataset.test_dataset, env_config.N_CLIENTS)))
            dataset = [ClientDataset(data_config, d) for d in dataset]
        if env_config and env_config.LEARNING_METHOD == LearningMethod.VFL:
            dataset = [ClientDataset(data_config, (
                SensorWrapper(DatasetWrapper(dataset.train_dataset, env_config.transformation), i), 
                SensorWrapper(DatasetWrapper(dataset.test_dataset, env_config.transformation), i))) for i in range(len(env_config.transformation))]
        return dataset

    @staticmethod
    def load_model(data_config: DataConfiguration, torch_config: TorchConfiguration, env_config: EnvConfiguration, member_config: MemberConfiguration) -> nn.Module:
        if data_config.DATASET == DataSetEnum.MNIST and env_config.LEARNING_METHOD == LearningMethod.CL: 
            return CentralMNISTModel().to(torch_config.DEVICE)
        elif data_config.DATASET == DataSetEnum.MNIST and env_config.LEARNING_METHOD == LearningMethod.SL and member_config.type == Member.SERVER:
            return SplitMNISTServerModel().to(torch_config.DEVICE)
        elif data_config.DATASET == DataSetEnum.MNIST and env_config.LEARNING_METHOD == LearningMethod.SL and member_config.type == Member.CLIENT:
            return SplitMNISTClientModel().to(torch_config.DEVICE)
        elif data_config.DATASET == DataSetEnum.MNIST and env_config.LEARNING_METHOD == LearningMethod.HFL: 
            return FederatedMNISTModel().to(torch_config.DEVICE)
        elif data_config.DATASET == DataSetEnum.MNIST and env_config.LEARNING_METHOD == LearningMethod.VFL and member_config.type == Member.SERVER: 
            return FederatedMNISTServerModel().to(torch_config.DEVICE)
        elif data_config.DATASET == DataSetEnum.MNIST and env_config.LEARNING_METHOD == LearningMethod.VFL and member_config.type == Member.CLIENT: 
            return FederatedMNISTClientModel().to(torch_config.DEVICE)


    @staticmethod
    def load_optimizer(model: nn.Module, model_config: ModelConfiguration):
        return optim.Adam(model.parameters(), lr=model_config.LEARNING_RATE)

    @staticmethod
    def load_criterion():
        return nn.CrossEntropyLoss()

    @staticmethod
    def load_aggregator(): 
        return ModelAggregator.model_avg

    @staticmethod
    def load_client_selector(): 
        return ClientSelector.random_selector

    
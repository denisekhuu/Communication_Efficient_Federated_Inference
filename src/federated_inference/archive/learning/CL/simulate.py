from federated_inference.configs.data_config import DataConfiguration
from federated_inference.configs.torch_config import TorchConfiguration
from federated_inference.configs.env_config import EnvConfiguration
from federated_inference.configs.model_config import CLModelConfiguration, ModelConfiguration
from federated_inference.configs.member_config import MemberConfiguration
from federated_inference.common.environment import Member
from federated_inference.common.env_utils import EnvUtils
from .client import CLClient 
from .server import CLServer
from torch.utils.data import DataLoader, Subset
from collections.abc import Iterable
from torchsummary import summary
from federated_inference.common.cost_calculator import CostCalculator
import torch
import math

def run(data_config: DataConfiguration, torch_config: TorchConfiguration, env_config: EnvConfiguration, model_config: ModelConfiguration, env_utils: EnvUtils, save: bool = True):
    dataset = env_utils.load_data(data_config, env_config)
    client = CLClient(dataset, data_config, torch_config, env_config, model_config, MemberConfiguration(member_type=Member.CLIENT), env_utils)
    server = CLServer(data_config, torch_config, env_config, model_config, MemberConfiguration(member_type=Member.SERVER, input_size=data_config.INSTANCE_SIZE), env_utils)
    data = client.send_all()
    server.run_training(data)
    if save:
        server.save()

    data, label = client.request_pred(idx=1, keep_label=True)
    print(server.pred(data), label)

if __name__ == "__main__":
    data_config = DataConfiguration()
    torch_config = TorchConfiguration()
    env_config = EnvConfiguration()
    model_config = CLModelConfiguration()
    dataset = env_utils.load_data(data_config, env_config)
    client = CLClient(dataset, data_config, torch_config, env_config, model_config, MemberConfiguration(member_type=Member.CLIENT), EnvUtils)
    server = CLServer(data_config, torch_config, env_config, model_config, MemberConfiguration(member_type=Member.SERVER, input_size=data_config.INSTANCE_SIZE), EnvUtils)
    data = client.send_all()
    server.run(data)
    server.save()
    data, label = client.request_pred(idx=1, keep_label=True)
    print(server.pred(data), label)
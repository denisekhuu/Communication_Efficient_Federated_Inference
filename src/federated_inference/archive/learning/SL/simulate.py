from federated_inference.configs.data_config import DataConfiguration
from federated_inference.configs.torch_config import TorchConfiguration
from federated_inference.configs.env_config import EnvConfiguration
from federated_inference.configs.model_config import ModelConfiguration
from federated_inference.common.environment import Member
from federated_inference.configs.member_config import MemberConfiguration
from federated_inference.common.env_utils import EnvUtils
from .client import SLClient 
from .server import SLServer
import torch

def run(data_config: DataConfiguration, torch_config: TorchConfiguration, env_config: EnvConfiguration, model_config: ModelConfiguration, env_utils: EnvUtils, save: bool = True):
    dataset = env_utils.load_data(data_config, env_config)
    client = SLClient(dataset, data_config, torch_config, env_config, model_config, MemberConfiguration(member_type=Member.CLIENT, input_size=data_config.INSTANCE_SIZE), env_utils)
    server = SLServer(data_config, torch_config, env_config, model_config, MemberConfiguration(member_type=Member.SERVER, input_size=(1, 320)), env_utils)
    client.send_train_request(server)
    if save:
        server.save()
        client.save()

    data, label = client.request_pred(idx=1, keep_label=True)
    print(server.pred(data), label)


if __name__ == "__main__": 
    from federated_inference.configs.model_config import SLModelConfiguration
    data_config = DataConfiguration()
    torch_config = TorchConfiguration()
    env_config = EnvConfiguration(learning_method=LearningMethod.SL)
    model_config = SLModelConfiguration()
    dataset = env_utils.load_data(data_config, env_config)
    client = SLClient(dataset, data_config, torch_config, env_config, model_config, MemberConfiguration(member_type=Member.CLIENT, input_size=data_config.INSTANCE_SIZE), EnvUtils)
    server = SLServer(data_config, torch_config, env_config, model_config, MemberConfiguration(member_type=Member.SERVER, input_size=(1, 320)), EnvUtils)
    client.send_train_request(server) 
    data, label = client.request_pred(idx=1, keep_label=True)
    print(server.pred(data), label)
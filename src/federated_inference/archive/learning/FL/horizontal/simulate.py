from federated_inference.configs.data_config import DataConfiguration
from federated_inference.configs.torch_config import TorchConfiguration
from federated_inference.configs.env_config import EnvConfiguration
from federated_inference.configs.model_config import ModelConfiguration
from federated_inference.common.environment import LearningMethod, DataDistributionType, Member
from federated_inference.configs.member_config import MemberConfiguration, HFLMemberConfiguration
from federated_inference.common.env_utils import EnvUtils
from .client import HFLClient
from .server import HFLServer


def run(data_config: DataConfiguration, torch_config: TorchConfiguration, env_config: EnvConfiguration, model_config: ModelConfiguration, env_utils: EnvUtils, save: bool = True):
    datasets = EnvUtils.load_data(data_config, env_config)
    clients = [HFLClient(dataset, data_config, torch_config, env_config, model_config, HFLMemberConfiguration(idx = idx, member_type=Member.CLIENT, input_size=data_config.INSTANCE_SIZE), EnvUtils) for idx, dataset in enumerate(datasets)]
    server = HFLServer(data_config, torch_config, env_config, model_config, HFLMemberConfiguration(member_type=Member.SERVER, input_size=data_config.INSTANCE_SIZE), EnvUtils)
    for client in clients: 
        server.register_client(client)
    server.train()
    if save: 
        server.save()

if __name__ == "__main__":
    from federated_inference.configs.env_config import FederatedEnvConfiguration     
    from federated_inference.configs.model_config import HFLModelConfiguration   
    from federated_inference.configs.member_config import HFLMemberConfiguration
    data_config = DataConfiguration()
    torch_config = TorchConfiguration()
    env_config = FederatedEnvConfiguration(learning_method=LearningMethod.HFL)
    model_config = HFLModelConfiguration(4)
    datasets = EnvUtils.load_data(data_config, env_config)
    clients = [HFLClient(dataset, data_config, torch_config, env_config, model_config, HFLMemberConfiguration(idx = idx, member_type=Member.CLIENT, input_size=data_config.INSTANCE_SIZE), EnvUtils) for idx, dataset in enumerate(datasets)]
    server = HFLServer(data_config, torch_config, env_config, model_config, HFLMemberConfiguration(member_type=Member.SERVER, input_size=data_config.INSTANCE_SIZE), EnvUtils)
    for client in clients: 
        server.register_client(client)
    server.train()

    
    from federated_inference.configs.data_config import DataConfiguration
    from federated_inference.configs.torch_config import TorchConfiguration
    from federated_inference.configs.env_config import EnvConfiguration, FederatedEnvConfiguration
    from federated_inference.configs.model_config import ModelConfiguration, SLModelConfiguration, CLModelConfiguration, HFLModelConfiguration, VFLModelConfiguration
    from federated_inference.common.environment import LearningMethod, DataDistributionType, Member
    from federated_inference.configs.member_config import MemberConfiguration, HFLMemberConfiguration
    from federated_inference.common.env_utils import EnvUtils

    data_config = DataConfiguration()
    torch_config = TorchConfiguration()
    
    from federated_inference.configs.env_config import FederatedEnvConfiguration     

    env_config = FederatedEnvConfiguration(learning_method=LearningMethod.VFL)
    model_config = VFLModelConfiguration(3)
    datasets = EnvUtils.load_data(data_config, env_config)
    clients = [VFLClient(dataset, data_config, torch_config, env_config, model_config, MemberConfiguration(idx = idx, member_type=Member.CLIENT, input_size=(1, 14,14)), EnvUtils) for idx, dataset in enumerate(datasets)]
    server = VFLServer(data_config, torch_config, env_config, model_config, MemberConfiguration(member_type=Member.SERVER, input_size=(1, 500)), EnvUtils)
    for client in clients: 
        server.register_client(client)
        client.register_server(server)

    clients[0].request_training()
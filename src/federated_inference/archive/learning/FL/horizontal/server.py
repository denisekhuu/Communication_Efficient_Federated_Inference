from torch.utils.data import DataLoader, Subset, Dataset
from torchsummary import summary
import torch
import math
from federated_inference.configs.data_config import DataConfiguration
from federated_inference.configs.torch_config import TorchConfiguration
from federated_inference.configs.env_config import EnvConfiguration
from federated_inference.configs.model_config import ModelConfiguration
from federated_inference.common.environment import LearningMethod, DataDistributionType, Member
from federated_inference.configs.member_config import MemberConfiguration
from federated_inference.common.env_utils import EnvUtils
from collections.abc import Iterable
import copy
from .client import HFLClient
import asyncio

class HFLServer():

    def __init__(self, 
            data_config: DataConfiguration, 
            torch_config: TorchConfiguration, 
            env_config: EnvConfiguration, 
            model_config: ModelConfiguration, 
            member_config: MemberConfiguration,
            env_utils: EnvUtils,
            model_summary: bool = True, 
            log: bool = True, log_interval: int = 20
        ):
            self.data_config = data_config
            self.torch_config = torch_config
            self.env_config = env_config
            self.model_config = model_config
            self.member_config = member_config
            self.model = env_utils.load_model(data_config, torch_config, env_config, member_config)
            
            self.client_conn = []
            self.aggregator = env_utils.load_aggregator()
            self.selector = env_utils.load_client_selector()

            self.log = log
            self.log_interval = log_interval
            if model_summary and member_config.input_size: 
                summary(self.model, member_config.input_size)

    def register_client(self, client: HFLClient):
        self.client_conn.append(client)

    async def request_learning_round(self): 
        selected_clients = self.selector(len(self.client_conn), self.member_config.CLIENTS_PER_ROUND)
        tasks = [self.client_conn[i].receive_train_request(self.model.state_dict()) for i in selected_clients]
        client_models = await asyncio.gather(*tasks)
        return self.aggregator(client_models)

    def train(self): 
        for round in range(self.member_config.N_ROUNDS):
            model = asyncio.run(self.request_learning_round())
            self.update_model(model)
            asyncio.run(self.broadcast_model())        

    async def broadcast_model(self):
        tasks = [client.update_model(self.model.state_dict()) for i, client in enumerate(self.client_conn)]
        await asyncio.gather(*tasks)
    
    def update_model(self, new_params):
        self.model.load_state_dict(copy.deepcopy(new_params), strict=True)
        self.model.eval()

    def save(self):
        import uuid
        import os
        idx = uuid.uuid4()
        path = os.path.join(self.env_config.RESULT_PATH, 'hfl', str(idx))
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, f'model_hfl_server.pth').replace("\\", "/")
        optimizer_path = os.path.join(path, f'optimizer_hfl_server.pth').replace("\\", "/")
        torch.save(self.model.state_dict(), model_path)
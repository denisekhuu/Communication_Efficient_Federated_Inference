from torch.utils.data import Dataset, Subset, DataLoader
from federated_inference.configs.data_config import DataConfiguration
from federated_inference.configs.torch_config import TorchConfiguration
from federated_inference.configs.env_config import EnvConfiguration
from federated_inference.configs.model_config import ModelConfiguration
from federated_inference.configs.member_config import MemberConfiguration
from federated_inference.common.env_utils import EnvUtils
from torchsummary import summary
import torch 
import asyncio
from collections.abc import Iterable
import itertools

class VFLClient(): 
    def __init__(self, 
        dataset: Dataset,
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
        self.data = dataset
        self.model = env_utils.load_model(data_config, torch_config, env_config, member_config)
        self.optimizer = env_utils.load_optimizer(self.model, model_config)
        self.criterion = env_utils.load_criterion()
        self._to_loader()
        self.log = log
        self.log_interval = log_interval

        if self.log: 
            self.train_losses = []
            self.val_losses = []
            self.accuracies = []

        if model_summary and member_config.input_size: 
            summary(self.model, member_config.input_size)

    def register_server(self, server):
        self.server = server

    def _to_loader(self):
        trainset = self.data.train_dataset
        traindata = Subset(trainset, range(round(self.model_config.TRAIN_RATIO*len(trainset))))
        valdata = Subset(trainset, range(round(self.model_config.TRAIN_RATIO*len(trainset)), len(trainset)))
        self.trainloader = DataLoader(traindata, batch_size=self.model_config.BATCH_SIZE_TRAIN, shuffle=self.model_config.TRAIN_SHUFFLE)
        self.valloader = DataLoader(valdata, batch_size=self.model_config.BATCH_SIZE_VAL, shuffle=self.model_config.VAL_SHUFFLE)

    def select_subset(self, ids: Iterable[int], set_type: str = "train"):
        if set_type == "test":
            return Subset(self.data.test_dataset, ids)
        else: 
            return Subset(self.data.train_dataset, ids)

    def request_training(self):
        for epoch in range(1, self.model_config.N_EPOCHS + 1):
            for batch_idx, (_, target) in enumerate(self.trainloader):
                target = target.to(self.torch_config.DEVICE).long()
                asyncio.run(self.server.request_learning_round(epoch, batch_idx, target))

        
    async def receive_train_request(self, epoch, batch_index):
        self.model.train()
        data, target = list(itertools.islice(self.trainloader, batch_index, batch_index + 1))[0]
        data = data.to(self.torch_config.DEVICE).float()
        target = target.to(self.torch_config.DEVICE).long()
        self.optimizer.zero_grad()
        activation = self.model(data)
        return activation

    async def backprop(self, epoch, batch_idx, activation, loss, activation_grad):
        activation.requires_grad_()
        activation.backward(activation_grad)
        self.optimizer.step()
        if self.log and batch_idx % self.log_interval == 0:
            self.train_losses.append(loss)

    def save(self):
        import uuid
        import os
        idx = uuid.uuid4()
        path = os.path.join(self.env_config.RESULT_PATH, 'sl')
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, f'model_sl_client_{idx}.pth').replace("\\", "/")
        optimizer_path = os.path.join(path, f'optimizer_sl_client_{idx}.pth').replace("\\", "/")
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)

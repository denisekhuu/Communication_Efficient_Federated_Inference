from federated_inference.configs.data_config import DataConfiguration
from federated_inference.configs.torch_config import TorchConfiguration
from federated_inference.configs.env_config import EnvConfiguration
from federated_inference.configs.model_config import ModelConfiguration
from federated_inference.configs.member_config import MemberConfiguration
from federated_inference.common.env_utils import EnvUtils
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
import torch
import math

class CLServer(): 

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
        self.optimizer = env_utils.load_optimizer(self.model, model_config)
        self.criterion = env_utils.load_criterion()

        self.log = log
        self.log_interval = log_interval

        if self.log: 
            self.train_losses = []
            self.val_losses = []
            self.accuracies = []

        if model_summary and member_config.input_size: 
            summary(self.model, member_config.input_size)

    def _to_loader(self, trainset):
        traindata = Subset(trainset, range(round(self.model_config.TRAIN_RATIO*len(trainset))))
        valdata = Subset(trainset, range(round(self.model_config.TRAIN_RATIO*len(trainset)), len(trainset)))
        self.trainloader = DataLoader(traindata, batch_size=self.model_config.BATCH_SIZE_TRAIN, shuffle=self.model_config.TRAIN_SHUFFLE)
        self.valloader = DataLoader(valdata, batch_size=self.model_config.BATCH_SIZE_VAL, shuffle=self.model_config.VAL_SHUFFLE)

    def train(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.trainloader):
            data = data.to(self.torch_config.DEVICE).float()
            target = target.to(self.torch_config.DEVICE).long()

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            if self.log and batch_idx % self.log_interval == 0:
                self.train_losses.append(loss.item())
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.trainloader.dataset)} '
                    f'({100. * batch_idx / len(self.trainloader):.0f}%)]\tLoss: {loss.item():.6f}')

    def validate(self):
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.valloader:
                data = data.to(self.torch_config.DEVICE).float()
                target = target.to(self.torch_config.DEVICE).long()
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.valloader.dataset)
        accuracy = 100. * correct / len(self.valloader.dataset)
        if self.log: 
            self.val_losses.append(test_loss)
            self.accuracies.append(accuracy)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.valloader.dataset)} '
            f'({accuracy:.0f}%)\n')
            
    def save(self):
        import uuid
        import os
        idx = uuid.uuid4()
        os.makedirs(self.env_config.RESULT_PATH, exist_ok=True)
        model_path = os.path.join(self.env_config.RESULT_PATH, f'model_cl_server_{idx}.pth').replace("\\", "/")
        optimizer_path = os.path.join(self.env_config.RESULT_PATH, f'optimizer_cl_server_{idx}.pth').replace("\\", "/")
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)

    def run_training(self, trainset):
        self._to_loader(trainset)
        self.validate()
        for epoch in range(1, self.model_config.N_EPOCH + 1):
            self.train(epoch)
            self.validate()

    def pred(self, data):
        data = data.to(self.torch_config.DEVICE).float()
        output = self.model(data)
        pred = output.argmax(dim=1, keepdim=True)
        return pred

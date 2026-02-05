from federated_inference.configs.data_config import DataConfiguration
from federated_inference.configs.torch_config import TorchConfiguration
from federated_inference.configs.env_config import EnvConfiguration
from federated_inference.configs.model_config import ModelConfiguration
from federated_inference.configs.member_config import MemberConfiguration
from federated_inference.common.env_utils import EnvUtils
from torch.utils.data import Dataset, DataLoader, Subset
from federated_inference.common.cost_calculator import CostCalculator
from collections.abc import Iterable
from torchsummary import summary
import torch 

class SLClient():
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
        self.log = log
        self.log_interval = log_interval

        if self.log: 
            self.train_losses = []
            self.val_losses = []
            self.accuracies = []

        if model_summary and member_config.input_size: 
            summary(self.model, member_config.input_size)

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

    def send_train_request(self, server):
        self._to_loader()
        self.validate(server)
        for epoch in range(1, self.model_config.N_EPOCHS + 1):
            self.train(server, epoch)
            self.validate(server)

    def train(self, server, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.trainloader):
            data = data.to(self.torch_config.DEVICE).float()
            target = target.to(self.torch_config.DEVICE).long()
            self.optimizer.zero_grad()
            server.optimizer.zero_grad()
            activation = self.model(data)
            loss, activation_grad = server.train(activation, target)
            activation.backward(activation_grad)
            self.optimizer.step()
            if self.log and batch_idx % self.log_interval == 0:
                self.train_losses.append(loss)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.trainloader.dataset),
                    100. * batch_idx / len(self.trainloader), loss))
                    

    def validate(self, server):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.valloader:
                data = data.to(self.torch_config.DEVICE).float()
                target = target.to(self.torch_config.DEVICE).long()

                activation = self.model(data)
                output = server.forward(activation)
                test_loss += self.criterion(output, target).item()

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.valloader.dataset)
        accuracy = 100. * correct / len(self.valloader.dataset)
        if self.log: 
            self.val_losses.append(test_loss)
            self.accuracies.append(accuracy)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.valloader.dataset), accuracy))

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


    def request_pred(self, idx: int|None = None, set_type: str = "test", pred_all: bool= False, keep_label: bool = False): 
        if idx != None:
            if set_type == "test":
                data = self.data.test_dataset[idx]
                img = data[0].to(self.torch_config.DEVICE).float()
                activation = self.model(img)
                CostCalculator.calculate_communication_cost_by(activation)
                return (activation, data[1]) if keep_label else activation
        elif pred_all:
            CostCalculator.calculate_communication_cost_by(self.data.test_dataset)
            data = [(self.model(img.to(self.torch_config.DEVICE).float()), label) for img, label in self.data.test_dataset]
            return data if keep_label else [img for img, label in data]


from federated_inference.configs.data_config import DataConfiguration
from federated_inference.configs.torch_config import TorchConfiguration
from federated_inference.configs.env_config import EnvConfiguration
from federated_inference.configs.model_config import ModelConfiguration
from federated_inference.configs.member_config import MemberConfiguration
from federated_inference.common.env_utils import EnvUtils
from torchsummary import summary
import torch 

class SLServer():
    def __init__(
        self, 
        data_config: DataConfiguration, 
        torch_config: TorchConfiguration, 
        env_config: EnvConfiguration, 
        model_config: ModelConfiguration, 
        member_config: MemberConfiguration,
        env_utils: EnvUtils,
        model_summary: bool = True, 
        log: bool = True, log_interval: int = 20):
        
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

        if model_summary and member_config.input_size: 
            summary(self.model, member_config.input_size)

    def train(self, activation, target):
        self.model.train()
        activation.requires_grad_()
        output = self.model(activation)
        loss = self.criterion(output, target)
        if self.log: 
            self.train_losses.append(loss.item())
        # Get gradients for both activation and model parameters
        grads = torch.autograd.grad(loss, [activation] + list(self.model.parameters()))
        activation_grad = grads[0]
        model_grads = grads[1:]
        # Apply gradients manually to model parameters
        for param, grad in zip(self.model.parameters(), model_grads):
            param.grad = grad  # Set .grad for optimizer

        self.optimizer.step()
        return loss.item(), activation_grad

    def forward(self, data):
        self.model.eval()
        data = data.to(self.torch_config.DEVICE).float()
        return self.model(data)

    def pred(self, data):
        data = data.to(self.torch_config.DEVICE).float()
        output = self.model(data)
        pred = output.argmax(dim=1, keepdim=True)
        return pred

    def save(self):
        import uuid
        import os
        idx = uuid.uuid4()
        path = os.path.join(self.env_config.RESULT_PATH, 'sl')
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, f'model_sl_server_{idx}.pth').replace("\\", "/")
        optimizer_path = os.path.join(path, f'optimizer_sl_server_{idx}.pth').replace("\\", "/")
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)
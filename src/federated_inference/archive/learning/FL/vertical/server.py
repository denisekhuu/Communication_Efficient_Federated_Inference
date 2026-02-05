class VFLServer():
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
        self.client_conn = []
        self.model = env_utils.load_model(data_config, torch_config, env_config, member_config)
        self.optimizer = env_utils.load_optimizer(self.model, model_config)
        self.criterion = env_utils.load_criterion()

        self.log = log
        self.log_interval = log_interval

        if self.log: 
            self.train_losses = []

        if model_summary and member_config.input_size: 
            summary(self.model, member_config.input_size)

    def register_client(self, client: VFLClient):
        self.client_conn.append(client)

    async def request_learning_round(self, epoch, batch_idx, target): 
        activation_tasks = [client.receive_train_request(epoch, batch_idx) for i in self.client_conn]
        client_activations = await asyncio.gather(*activation_tasks)
        concat_activation = torch.cat(client_activations, dim=1)
        loss, concat_activation_grad = self.train(epoch, batch_idx, concat_activation, target)
        activation_sizes = [act.shape[1] for act in client_activations]
        activation_grads = torch.split(concat_activation_grad, activation_sizes, dim=1)
        tasks = [client.backprop(epoch, batch_idx, activation, loss, activation_grad) for client, activation, activation_grad in list(zip(self.client_conn, client_activations, activation_grads))]
        await asyncio.gather(*tasks)


    def train(self, epoch, batch_idx, activation, target):
        self.model.train()
        self.optimizer.zero_grad()
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
        if self.log and batch_idx % self.log_interval == 0:
            print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss))
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
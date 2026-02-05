import torch.nn as nn
import torch 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from federated_inference.dataset import MNISTDataset
from federated_inference.configs.data_config import DataConfiguration
from torchsummary import summary

class ClientCNN(nn.Module):
    def __init__(self):
        super(ClientCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x

class ServerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2000, 50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, 10)
        )

    def forward(self, x_concat):
        return self.classifier(x_concat)

# ------------------------------
# Training and Testing Functions
# ------------------------------
def train(client_models, server_model, clients, criterion, client_optimizers, server_optimizer, epoch):
    for batch_idx, batches in enumerate(zip(*[client.trainloader for client in clients])):
        data_slices = [batch[0].float().cuda() for batch in batches]
        target = batches[0][1].long().cuda()

        activations = []
        for data, client_model in zip(data_slices, client_models):
            activation = client_model(data)
            activation.requires_grad_()
            activations.append(activation)

        concat_activation = torch.cat(activations, dim=1)
        output = server_model(concat_activation)
        loss = criterion(output, target)

        # Backward pass
        for opt in client_optimizers:
            opt.zero_grad()
        server_optimizer.zero_grad()
        grads = torch.autograd.grad(loss, [activation] + list(server_model.parameters()))
        activation_grad = grads[0]
        model_grads = grads[1:]
        # Apply gradients manually to model parameters
        for param, grad in zip(server_model.parameters(), model_grads):
            param.grad = grad  
        server_optimizer.step()

        for opt in client_optimizers:
            opt.step()
        server_optimizer.step()

        if batch_idx % 20 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(target)}/{len(clients[0].trainloader.dataset)}] Loss: {loss.item():.4f}")


def test(client_models, server_model, clients, criterion):
    for model in client_models:
        model.eval()
    server_model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batches in zip(*[client.testloader for client in clients]):
            data_slices = [batch[0].float().cuda() for batch in batches]
            target = batches[0][1].long().cuda()

            activations = [model(data) for data, model in zip(data_slices, client_models)]
            concat_activation = torch.cat(activations, dim=1)

            output = server_model(concat_activation)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    total = len(clients[0].testloader.dataset)
    print(f"\nTest: Avg loss: {test_loss / total:.4f}, Accuracy: {correct}/{total} ({100. * correct / total:.2f}%)\n")

class DataTransformConfig(): 
    tensor_size = [1,28,28]
    method_name = 'full'
    mask_size = [14,14]
    dimensions =  [1,2] 
    stride = 14
    n_position = None
    drop_p = None
    
class Client():
    def __init__(self, idx, trainset, testset):
        self.id = idx
        self.trainset = trainset
        self.testset = testset
        self.trainloader = DataLoader(self.trainset, batch_size=64)  
        self.testloader = DataLoader(self.testset, batch_size=64)   
# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    from federated_inference.dataset import MNISTDataset
    from federated_inference.configs.data_config import DataConfiguration
    from federated_inference.transform import StridePartitionTransform, DataSplitter, SensorWrapper
    from torch.utils.data import DataLoader

    configs = DataConfiguration()
    data_transform_config = DataTransformConfig()
    op_transform = StridePartitionTransform(data_transform_config)
    number_of_clients = len(op_transform)
    data = MNISTDataset(configs, [op_transform])

    clients = [
        Client(i, 
            SensorWrapper(data.train_dataset, i), 
            SensorWrapper(data.test_dataset, i)) 
        for i in range(number_of_clients)]

    client_models = [ClientCNN().cuda() for _ in range(4)]
    summary(ClientCNN().cuda(), (1, 14, 14))
    server_model = ServerModel().cuda()
    summary(ServerModel().cuda(), (1, 2000))
    client_optimizers = [torch.optim.Adam(model.parameters(), lr=0.001) for model in client_models]
    server_optimizer = torch.optim.Adam(server_model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()
    n_epochs = 8

    for epoch in range(1, n_epochs + 1):
        train(client_models, server_model, clients, criterion, client_optimizers, server_optimizer, epoch)
        test(client_models, server_model, clients, criterion)
    

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from federated_inference.dataset import MNISTDataset
from federated_inference.configs import DataConfiguration

# ------------------------------
# Define models
# ------------------------------
class ClientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 320)
        return x


class ServerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        return self.classifier(x)


# ------------------------------
# Training and Testing Functions
# ------------------------------
def train(client_model, server_model, train_loader, criterion, client_optimizer, server_optimizer, epoch):
    client_model.train()
    server_model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(torch.float32)
        target = target.to(torch.long)

        # Forward
        activation = client_model(data)
        activation.requires_grad_()
        output = server_model(activation)
        loss = criterion(output, target)

        # Backward
        client_optimizer.zero_grad()
        server_optimizer.zero_grad()

        activation_grad = torch.autograd.grad(loss, activation)[0] # gradients from output to activation 
        activation.backward(activation_grad) # gradients from output to activation 

        client_optimizer.step() # gradient update client
        server_optimizer.step() # gradient update server

        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(client_model, server_model, test_loader, criterion):
    client_model.eval()
    server_model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(torch.float32)
            target = target.to(torch.long)

            activation = client_model(data)
            output = server_model(activation)
            test_loss += criterion(output, target).item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    configs = DataConfiguration()
    mnist = MNISTDataset(configs)

    train_dataset = Subset(mnist.train_dataset, range(50000))
    test_dataset = mnist.test_dataset

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    client_model = ClientModel()
    server_model = ServerModel()

    criterion = nn.CrossEntropyLoss()
    client_optimizer = optim.Adam(client_model.parameters(), lr=0.001)
    server_optimizer = optim.Adam(server_model.parameters(), lr=0.001)

    n_epochs = 8
    for epoch in range(1, n_epochs + 1):
        train(client_model, server_model, train_loader, criterion, client_optimizer, server_optimizer, epoch)
        test(client_model, server_model, test_loader, criterion)

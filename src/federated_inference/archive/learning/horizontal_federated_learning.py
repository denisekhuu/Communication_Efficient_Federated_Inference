import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
from federated_inference.transform import DataSplitter
# client selection missing

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
        self.classifier = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 320)
        x = self.classifier(x)
        return x


# Aggregation function: averaging model weights
def aggregate_models(client_models: list, global_model) -> nn.Module:
    """
    Aggregates the models using the mean of each parameter's weights
    Args:
        client_models (list): List of client models
    Returns:
        nn.Module: Global model with averaged parameters
    """
    # Initialize the global model by copying the first client's model
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    global_dict = global_model.state_dict()

    # Initialize a dictionary to accumulate the weights
    for key in global_dict:
        global_dict[key] = torch.zeros_like(global_dict[key])

    # Loop through each client model and accumulate the weights
    for model in client_models:
        model_dict = model.state_dict()
        for key in model_dict:
            global_dict[key] += model_dict[key] / len(client_models)

    # Load the averaged weights into the global model
    global_model.load_state_dict(global_dict)

# Sync Client Models with Global Model
def sync_client_models_with_global(client_models: list, global_model: nn.Module):
    """
    Sync each client model with the global model's parameters.
    Args:
        client_models (list): List of client models
        global_model (nn.Module): Global model with the latest parameters
    """
    global_dict = global_model.state_dict()
    for client_model in client_models:
        client_model.load_state_dict(global_dict)  # Set client model's parameters to global model


# Example usage for client creation and federated learning
def run_federated_learning(dataset, global_model, batch_size=64, n_clients=3, epochs=5, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    # Create client models
    client_models = [ClientModel().to(device) for _ in range(n_clients)]

    # Define optimizers for each client
    optimizers = [optim.Adam(client_model.parameters(), lr=lr) for client_model in client_models]
    
    # Split the dataset IID among clients
    client_datasets = DataSplitter.divide_data_equally(dataset, n_clients)
    
    # Create DataLoaders for each client
    client_loaders = DataSplitter.create_dist_loaders(client_datasets, batch_size)

    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Sync all client models with the global model at the beginning of each epoch
        sync_client_models_with_global(client_models, global_model)

        # Train each client independently
        for client_id in range(n_clients):
            client_model = client_models[client_id]
            optimizer = optimizers[client_id]
            train_loader = client_loaders[client_id]

            client_model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for data, target in train_loader:
                # Move data to GPU if available
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                output = client_model(data)

                # Compute loss
                loss = nn.CrossEntropyLoss()(output, target)

                # Backward pass
                loss.backward()

                # Optimizer step
                optimizer.step()

                # Accumulate loss and accuracy
                running_loss += loss.item()
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)

            print(f"Client {client_id+1} - Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")

        # Aggregate models from all clients after each epoch
        aggregate_models(client_models, global_model)

        # Optionally, print out the global model's performance (for debugging)
        print(f"\nGlobal model aggregated and updated.")

    return global_model

# Testing function for the global model
def test_global_model(global_model, test_loader):
    """
    Tests the global model on the test dataset.
    Args:
        global_model (nn.Module): The global model to test
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset
    """
    global_model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    test_loss = 0.0

    criterion = nn.CrossEntropyLoss()  # Define the loss function
    
    with torch.no_grad():  # Disable gradient computation
        for data, target in test_loader:
            # Move data to GPU if available
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # Forward pass
            output = global_model(data)

            # Compute loss
            loss = criterion(output, target)
            test_loss += loss.item()

            # Get predictions
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    # Calculate and print test accuracy and loss
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')


# Example usage
if __name__ == "__main__":
    from torchvision import datasets, transforms
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # Create DataLoader for the test dataset
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    global_model = ClientModel().to(device)

    # Run Federated Learning with 3 clients
    run_federated_learning(train_dataset, global_model, batch_size=64, n_clients=3, epochs=5, lr=0.001)

    # Optionally, you can test the global model here after training is done.
    print("\nFinal Testing of Global Model:")
    test_global_model(global_model, test_loader)
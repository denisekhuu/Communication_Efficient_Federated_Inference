import torch.nn as nn

class FederatedMNISTClientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.Dropout(),
            nn.ReLU(),
            nn.Conv2d(10, 5, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.Dropout(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x
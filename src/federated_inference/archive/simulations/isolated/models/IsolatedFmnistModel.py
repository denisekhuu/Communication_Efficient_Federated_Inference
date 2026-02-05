import torch.nn as nn

class IsolatedFMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "SingleMNISTModel"
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=3),
            nn.Dropout(),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(180, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 180)
        x = self.classifier(x)
        return x

import torch.nn as nn
class NaiveMNISTModel(nn.Module):
    def __init__(self):
        self.name = "NaiveMNISTModel"
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(5,21), padding = (1,0)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=(5,11), padding = (2,0)),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(240, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 240)
        x = self.classifier(x)
        return x
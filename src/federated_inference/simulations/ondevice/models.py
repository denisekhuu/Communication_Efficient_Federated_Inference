import torch.nn as nn

class OnDeviceMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "OnDeviceMNISTModel"

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2), 
            nn.Dropout(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten dynamically
        x = self.classifier(x)
        return x
    

class OnDeviceCIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "OnDeviceCIFAR10Model"

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2), 
            nn.Dropout(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten dynamically
        x = self.classifier(x)
        return x
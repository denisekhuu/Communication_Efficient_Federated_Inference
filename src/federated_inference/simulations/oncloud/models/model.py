import torch.nn as nn

class OnCloudMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "OnCloudMNISTModel"
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # output: 32x14x56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # output: 64x7x28
            nn.Dropout(0.25)
        )


        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 28, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.5),
            nn.Linear(16, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class OnCloudCIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "OnCloudCIFAR10Model"

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),     # [B, 32, 14, 56]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # [B, 64, 14, 56]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                                # → [B, 64, 7, 28]
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # [B, 128, 7, 28]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                                # → [B, 128, 3, 14]
            nn.Dropout(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 14, 256),  # = 5376
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
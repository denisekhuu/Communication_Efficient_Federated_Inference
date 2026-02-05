import torch.nn as nn
class HybridSplitBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.deeper_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # [B, 128, 1, 1]
        )

        self.project = nn.Sequential(
            nn.Linear(128, 196)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.deeper_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.project(x) 
        return x
    


class LocalHybridSplitClassifierHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(196, 256*2),
            nn.BatchNorm1d(256*2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256*2, 10)
        )

    def forward(self, x):
        return self.classifier(x)

class RouterHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(196, 256*2),
            nn.BatchNorm1d(256*2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256*2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)

class GlobalHybridSplitClassifierHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(196*4, 256*2),
            nn.BatchNorm1d(256*2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256*2, 10),
        )

    def forward(self, x_concat):
        return self.classifier(x_concat)
        
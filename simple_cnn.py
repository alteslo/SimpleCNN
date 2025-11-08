import torch
import torch.nn as nn

from commutator import CommutatorConv2d


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, use_commutator=False):
        super().__init__()
        if use_commutator:
            self.conv1 = CommutatorConv2d(1, kernel_size=3, padding=1, adaptive=True)
        else:
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32 if not use_commutator else 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear((14 if use_commutator else 14 * 14 * 32), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        if isinstance(self.conv1, CommutatorConv2d):
            # после нашего слоя — 1 канал
            x = self.bn1(x)  # BatchNorm2d(1)
        else:
            x = self.bn1(x)  # BatchNorm2d(32)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

import torch
import torch.nn as nn

from commutator import CommutatorConv2d


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, use_commutator=False):
        super().__init__()
        self.use_commutator = use_commutator

        if use_commutator:
            # Наш слой → всегда 1 выходной канал
            self.conv1 = CommutatorConv2d(
                in_channels=1, kernel_size=3, padding=1, adaptive=True
            )
            self.bn1 = nn.BatchNorm2d(1)
            conv_out_channels = 1
        else:
            # Стандартная свёртка → 32 канала
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            conv_out_channels = 32

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)  # уменьшает 28x28 → 14x14

        # После пулинга: H=W=14
        self.fc_input_dim = conv_out_channels * 14 * 14
        self.fc = nn.Linear(self.fc_input_dim, num_classes)

    def forward(self, x):
        # x: [B, 1, 28, 28]
        x = self.conv1(x)  # [B, 1, 28, 28] или [B, 32, 28, 28]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # [B, C, 14, 14]
        x = torch.flatten(x, 1)  # [B, C*14*14]
        x = self.fc(x)  # [B, 10]
        return x
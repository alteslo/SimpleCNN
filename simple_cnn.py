import torch
import torch.nn as nn
from commutator import CommutatorConv2d


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, use_commutator=False, in_channels=1):
        super().__init__()
        self.use_commutator = use_commutator

        if use_commutator:
            self.conv1 = CommutatorConv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                padding=1,
                adaptive_weights=True,
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)  # всегда 32
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        # Размер после пулинга: 28 -> 14 (для Fashion-MNIST)
        # Для CIFAR-10: 32 -> 16
        self.fc_input_dim = 32 * 14 * 14  # под Fashion-MNIST
        # Если нужно универсально — вычислять динамически или параметризовать
        self.fc = nn.Linear(self.fc_input_dim, num_classes)

    def forward(self, x):
        x = self.conv1(x)  # [B, 32, 28, 28]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # [B, 32, 14, 14]
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

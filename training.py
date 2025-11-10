import torch
import torch.nn as nn
from torchvision import datasets, transforms

from simple_cnn1 import SimpleCNN

model = SimpleCNN(use_commutator=True)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# Загрузка данных (пример для Fashion-MNIST)

transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

# Обучение (1 эпоха для теста)
model.train()
for inputs, labels in trainloader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    # break  # для проверки
print("Обучение прошло успешно!")

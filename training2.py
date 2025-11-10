import torch
import torch.nn as nn
from torchvision import datasets, transforms
from simple_cnn1 import SimpleCNN

torch.cuda.set_per_process_memory_fraction(0.8)
device = torch.device("cuda:0")

# Инициализация модели и оптимизатора
model = SimpleCNN(use_commutator=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# Загрузка данных
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True)

# Обучение с GPU
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in trainloader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(trainloader)
    print(f"Эпоха [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("Обучение завершено!")

torch.save(model.state_dict(), "fashion_mnist_model.pth")
print("Модель сохранена!")

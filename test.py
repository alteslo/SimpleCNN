import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from simple_cnn import SimpleCNN

# Устройство
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# Загрузка обученной модели
model = SimpleCNN(use_commutator=True).to(device)
model.load_state_dict(torch.load("fashion_mnist_model.pth", map_location=device))
model.eval()
print("Модель загружена!")

# Загрузка тестовых данных
transform = transforms.Compose([transforms.ToTensor()])
testset = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# Классы Fashion-MNIST
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def evaluate_model(model, test_loader):
    """Полная оценка модели"""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    # Для матрицы ошибок
    confusion_matrix = torch.zeros(10, 10, dtype=torch.int32)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Обновляем матрицу ошибок
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    accuracy = 100 * correct / total
    return accuracy, all_predictions, all_labels, all_probabilities, confusion_matrix


def print_classification_report(true_labels, predictions, class_names):
    """Ручная реализация classification report"""
    num_classes = len(class_names)

    print("\nДЕТАЛЬНЫЙ ОТЧЕТ ПО КЛАССАМ:")
    print(
        f"{'Класс':<15} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Support':<10}"
    )
    print("-" * 55)

    for i in range(num_classes):
        # True Positive, False Positive, False Negative
        tp = sum((torch.tensor(predictions) == i) & (torch.tensor(true_labels) == i))
        fp = sum((torch.tensor(predictions) == i) & (torch.tensor(true_labels) != i))
        fn = sum((torch.tensor(predictions) != i) & (torch.tensor(true_labels) == i))

        # Метрики
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        support = sum(torch.tensor(true_labels) == i)

        print(
            f"{class_names[i]:<15} {precision:.3f}      {recall:.3f}      {f1:.3f}      {support:<10}"
        )


def plot_confusion_matrix(confusion_matrix, class_names):
    """Визуализация матрицы ошибок"""
    plt.figure(figsize=(10, 8))

    # Нормализуем для цветовой карты
    cm_normalized = confusion_matrix.float() / confusion_matrix.sum(
        dim=1, keepdim=True
    ).clamp(min=1)

    plt.imshow(cm_normalized.numpy(), cmap="Blues", interpolation="nearest")
    plt.colorbar()

    # Добавляем текстовые аннотации
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(
                j,
                i,
                f"{confusion_matrix[i, j].item()}\n({cm_normalized[i, j]:.2f})",
                ha="center",
                va="center",
                color="white" if cm_normalized[i, j] > 0.5 else "black",
                fontsize=8,
            )

    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.title("Матрица ошибок")
    plt.tight_layout()
    plt.show()


def analyze_confidence(probabilities, true_labels, predictions):
    """Анализ уверенности модели"""
    confidences = [probs[pred] for probs, pred in zip(probabilities, predictions)]
    correct_confidences = [
        conf for i, conf in enumerate(confidences) if predictions[i] == true_labels[i]
    ]
    wrong_confidences = [
        conf for i, conf in enumerate(confidences) if predictions[i] != true_labels[i]
    ]

    print(f"\nАНАЛИЗ УВЕРЕННОСТИ:")
    print(
        f"Средняя уверенность на правильных предсказаниях: {np.mean(correct_confidences):.3f}"
    )
    print(f"Средняя уверенность на ошибках: {np.mean(wrong_confidences):.3f}")
    print(f"Минимальная уверенность: {np.min(confidences):.3f}")
    print(f"Максимальная уверенность: {np.max(confidences):.3f}")

    # Гистограмма уверенности
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(correct_confidences, bins=20, alpha=0.7, label="Правильные", color="green")
    plt.xlabel("Уверенность")
    plt.ylabel("Частота")
    plt.title("Уверенность на правильных предсказаниях")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(wrong_confidences, bins=20, alpha=0.7, label="Ошибки", color="red")
    plt.xlabel("Уверенность")
    plt.ylabel("Частота")
    plt.title("Уверенность на ошибках")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Основная оценка
accuracy, predictions, true_labels, probabilities, cm = evaluate_model(
    model, testloader
)

print(f"\n{'=' * 50}")
print(f"ТОЧНОСТЬ НА ТЕСТОВОМ НАБОРЕ: {accuracy:.2f}%")
print(f"{'=' * 50}")

# Детальный отчет по классам
print_classification_report(true_labels, predictions, classes)

# Визуализация матрицы ошибок
plot_confusion_matrix(cm, classes)

# Анализ уверенности
analyze_confidence(probabilities, true_labels, predictions)


# Визуализация примеров
def visualize_predictions(model, test_loader, num_examples=8):
    """Визуализация предсказаний"""
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images[:num_examples])
        _, predictions = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(num_examples):
        axes[i].imshow(images[i].cpu().squeeze(), cmap="gray")
        pred_class = classes[predictions[i]]
        true_class = classes[labels[i]]
        confidence = probabilities[i][predictions[i]].item()

        title = f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.3f}"

        axes[i].set_title(title)
        axes[i].axis("off")

        # Подсветка ошибок красным
        if predictions[i] != labels[i]:
            axes[i].title.set_color("red")

    plt.tight_layout()
    plt.show()


print("\nВИЗУАЛИЗАЦИЯ ПРЕДСКАЗАНИЙ:")
visualize_predictions(model, testloader)


# Анализ по классам
def analyze_per_class_accuracy(confusion_matrix, class_names):
    """Анализ точности по классам"""
    print(f"\n{'=' * 50}")
    print("ТОЧНОСТЬ ПО КЛАССАМ:")
    print(f"{'=' * 50}")

    class_accuracy = torch.diag(confusion_matrix) / confusion_matrix.sum(dim=1).float()

    for i, class_name in enumerate(class_names):
        accuracy = class_accuracy[i].item() * 100
        support = confusion_matrix[i].sum().item()
        print(f"{class_name:<15}: {accuracy:6.2f}% ({support:>5} примеров)")

    # Визуализация точности по классам
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, class_accuracy.numpy() * 100, color="skyblue")
    plt.xticks(rotation=45)
    plt.ylabel("Точность (%)")
    plt.title("Точность по классам")
    plt.ylim(0, 100)

    # Добавляем значения на столбцы
    for bar, acc in zip(bars, class_accuracy.numpy() * 100):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


analyze_per_class_accuracy(cm, classes)

# Сравнение с базовыми показателями
print(f"\n{'=' * 50}")
print("СРАВНЕНИЕ С БАЗОВЫМИ ПОКАЗАТЕЛЯМИ:")+
print(f"{'=' * 50}")
print(f"Ваша модель с CommutatorConv2d: {accuracy:.2f}%")
print("Случайное угадывание: ~10.00%")
print("Простая линейная модель: ~80-85%")
print("Обычная CNN (32 канала): ~88-92%")
print("State-of-the-art: ~96-97%")

print(
    f"\nОЦЕНКА: Ваша модель показывает {'ХОРОШИЙ' if accuracy > 85 else 'УДОВЛЕТВОРИТЕЛЬНЫЙ'} результат!"
)
print(
    f"CommutatorConv2d {'работает эффективно' if accuracy > 80 else 'требует доработки'}."
)

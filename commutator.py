import torch
import torch.nn as nn
import torch.nn.functional as F


class CommutatorConv2d(nn.Module):
    def __init__(
        self, in_channels, kernel_size=3, stride=1, padding=0, adaptive=False, bias=True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.adaptive = adaptive

        # Обучаемое ядро K: [k, k]
        self.K = nn.Parameter(torch.randn(kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.K, a=0)  # инициализация как в стандартной свёртке

        # Смещение (bias)
        if bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("bias", None)

        # Обучаемые коэффициенты λc, λa (скаляры)
        if adaptive:
            self.lambda_c = nn.Parameter(torch.tensor(0.0))  # начальное значение: 0
            self.lambda_a = nn.Parameter(torch.tensor(1.0))  # начальное значение: 1
        else:
            # фиксированные значения (можно менять вручную)
            self.register_buffer("lambda_c", torch.tensor(0.0))
            self.register_buffer("lambda_a", torch.tensor(1.0))

    def forward(self, x):
        # x: [B, C_in, H, W]
        B, C, H, W = x.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        if p > 0:
            x = F.pad(x, (p, p, p, p))  # padding: (left, right, top, bottom)

        # Извлечение всех локальных блоков размером k×k
        # unfold возвращает: [B, C, k*k, L], где L — число позиций
        patches = x.unfold(2, k, s).unfold(3, k, s)  # [B, C, H_out, W_out, k, k]
        H_out, W_out = patches.shape[2], patches.shape[3]
        patches = patches.contiguous()  # для эффективного reshape

        # Преобразуем в [B, C, H_out * W_out, k, k]
        patches = patches.view(B, C, -1, k, k)  # [B, C, N, k, k], N = H_out*W_out

        # Ядро K: [k, k] → расширяем до [1, 1, 1, k, k] для бродкастинга
        K = self.K.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, k, k]

        # Матричные произведения: X @ K^T и K^T @ X
        # Используем torch.einsum для читаемости и контроля
        XK = torch.einsum("bcnij,jk->bcnik", patches, K.squeeze().T)  # [B, C, N, k, k]
        KX = torch.einsum("jk,bcnki->bcnji", K.squeeze().T, patches)  # [B, C, N, k, k]

        # Коммутатор и антикоммутатор
        commutator = XK - KX  # [B, C, N, k, k]
        anticommutator = XK + KX  # [B, C, N, k, k]

        # Агрегация: сумма всех элементов матрицы → скаляр на блок
        comm_sum = commutator.sum(dim=(-2, -1))  # [B, C, N]
        anti_sum = anticommutator.sum(dim=(-2, -1))  # [B, C, N]

        # Комбинируем с λ и суммируем по каналам
        out = self.lambda_c * comm_sum + self.lambda_a * anti_sum  # [B, C, N]
        out = out.sum(dim=1)  # сумма по входным каналам → [B, N]

        # Добавляем смещение
        if self.bias is not None:
            out = out + self.bias

        # Возвращаем в форму признаковой карты: [B, 1, H_out, W_out]
        out = out.view(B, 1, H_out, W_out)
        return out

# Создаём слой
layer = CommutatorConv2d(
    in_channels=1, kernel_size=3, stride=1, padding=1, adaptive=True
)

# Пример входа: один канал, изображение 28×28
x = torch.randn(4, 1, 28, 28)  # batch=4

# Прямой проход
y = layer(x)
print("Вход:", x.shape)  # [4, 1, 28, 28]
print("Выход:", y.shape)  # [4, 1, 28, 28]

# Проверим, что градиенты проходят
y.sum().backward()
print("λ_c.grad =", layer.lambda_c.grad)
print("K.grad =", layer.K.grad is not None)
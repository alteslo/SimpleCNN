import torch
import torch.nn as nn
import torch.nn.functional as F


class CommutatorConv2d(nn.Module):
    """
    Альтернативный сверточный слой на основе матричного коммутатора и антикоммутатора.

    Параметры:
        in_channels (int): количество входных каналов.
        out_channels (int): количество выходных каналов.
        kernel_size (int): размер ядра (только нечётные значения, рекомендуется 3).
        stride (int): шаг свёртки.
        padding (int): тип заполнения границ.
        adaptive_weights (bool): если True, λ_c и λ_a — обучаемые параметры.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        adaptive_weights=False,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "Ядро должно быть нечётного размера"
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Обучаемые веса фильтров: [out_channels, in_channels, k, k]
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        # Коэффициенты вклада коммутатора и антикоммутатора
        if adaptive_weights:
            # Начальная инициализация: λ_c = 0, λ_a = 1 (поведение как у классической свёртки)
            self.lambda_c = nn.Parameter(torch.tensor(0.0))
            self.lambda_a = nn.Parameter(torch.tensor(1.0))
        else:
            self.lambda_c = 0.0
            self.lambda_a = 1.0

        self.reset_parameters()

    def reset_parameters(self):
        # Инициализация весов по Каймину Хе (He initialization)
        nn.init.kaiming_uniform_(self.weight, a=0, mode="fan_in", nonlinearity="relu")
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / torch.sqrt(torch.tensor(fan_in))
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Вход: x ∈ R^(B, C_in, H, W)
        Выход: y ∈ R^(B, C_out, H_out, W_out)
        """
        B, C_in, H, W = x.shape
        k = self.kernel_size
        pad = self.padding
        stride = self.stride

        # Применение padding
        if pad > 0:
            x_padded = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0)
        else:
            x_padded = x

        H_out = (H + 2 * pad - k) // stride + 1
        W_out = (W + 2 * pad - k) // stride + 1

        # Инициализация выходного тензора
        y = torch.zeros(
            B, self.out_channels, H_out, W_out, device=x.device, dtype=x.dtype
        )

        # Итерация по позициям ядра (можно векторизовать, но для ясности оставлено явно)
        for i in range(H_out):
            for j in range(W_out):
                # Извлечение локального блока: [B, C_in, k, k]
                h_start, h_end = i * stride, i * stride + k
                w_start, w_end = j * stride, j * stride + k
                X_block = x_padded[
                    :, :, h_start:h_end, w_start:w_end
                ]  # [B, C_in, k, k]

                # Для каждого выходного канала
                for out_ch in range(self.out_channels):
                    K = self.weight[out_ch]  # [C_in, k, k]

                    # Компоненты по каждому входному каналу
                    comm_sum = 0.0
                    anticom_sum = 0.0

                    for in_ch in range(C_in):
                        X_ch = X_block[:, in_ch, :, :]  # [B, k, k]
                        K_ch = K[in_ch, :, :]  # [k, k]

                        # Матричные произведения
                        XK = torch.matmul(X_ch, K_ch)  # [B, k, k]
                        KX = torch.matmul(K_ch, X_ch)  # [B, k, k]

                        comm = XK - KX  # коммутатор
                        anticom = XK + KX  # антикоммутатор

                        # Агрегация: сумма всех элементов (1^T M 1)
                        comm_sum += comm.sum(dim=(1, 2))  # [B]
                        anticom_sum += anticom.sum(dim=(1, 2))  # [B]

                    # Линейная комбинация
                    activation = (
                        self.lambda_c * comm_sum
                        + self.lambda_a * anticom_sum
                        + self.bias[out_ch]
                    )  # [B]

                    y[:, out_ch, i, j] = activation

        return y

import torch

print("=== Информация о CUDA ===")
print(f"Доступен CUDA: {torch.cuda.is_available()}")
print(f"Количество GPU: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Всего памяти: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Используется: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")

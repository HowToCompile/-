import torch
print("CUDA 是否可用:", torch.cuda.is_available())
print("当前 GPU 数量:", torch.cuda.device_count())
print("GPU 设备名称:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无 GPU")

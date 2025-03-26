import torch

# CUDA 초기화
torch.cuda.init()

print(torch.cuda.is_available())  # True이면 CUDA 사용 가능
print(torch.cuda.current_device())  # 현재 사용 중인 GPU 번호
print(torch.cuda.get_device_name(0))  # GPU 0의 이름 확인
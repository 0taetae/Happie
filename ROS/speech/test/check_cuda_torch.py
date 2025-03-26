import torch

print(torch.__version__)

# CUDA가 사용 가능한지 확인
print(torch.cuda.is_available())  # True이면 CUDA 사용 가능

if torch.cuda.is_available():
    print(torch.cuda.current_device())  # 현재 사용 중인 GPU 번호
    print(torch.cuda.get_device_name(0))  # GPU 0의 이름 확인
else:
    print("CUDA가 사용 불가능합니다.")

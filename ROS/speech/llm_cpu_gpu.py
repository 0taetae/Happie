import re
from llama_cpp import Llama
import os
import torch

# .gguf 모델 파일 경로 설정
model_path = r"C:\Users\SSAFY\Desktop\LLM\llama-3-Korean-Bllossom-8B-Q4_K_M.gguf"

# 경로가 존재하는지 확인
if not os.path.exists(model_path):
    raise ValueError(f"Model path does not exist: {model_path}")

# CUDA 환경 변수 설정 (GPU 사용)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 0 사용하도록 설정

# GPU 상태 확인
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Switching to CPU...")

# 디바이스 설정: GPU를 우선 사용하고, 실패하면 CPU로 자동 전환
device = "cuda"  # 기본적으로 GPU 사용
try:
    # GPU에서 모델을 로드 (메모리 부족 시 CPU로 자동 전환)
    llama = Llama(model_path=model_path, device=device)
    # print("Using GPU for model inference.")
except ValueError as e:
    # print(f"Error loading model with GPU: {e}")
    # print("Switching to CPU...")
    llama = Llama(model_path=model_path, device="cpu")  # GPU 에러 발생 시 CPU로 모델 로드
 
# 입력 텍스트 설정
input_text = "삼성병원에 대해 알려줘."

# 모델로 텍스트 생성 (max_tokens를 늘려줌)
output = llama(input_text, max_tokens=100)

# 출력 텍스트 처리: 문장 끝을 기준으로 분리
if isinstance(output, dict):
    output_text = output.get('choices', [{}])[0].get('text', 'No text generated')
else:
    output_text = output

# 문장 끝을 기준으로 끊기
# 한국어에서는 마침표, 물음표, 느낌표 등을 기준으로 문장을 나눌 수 있습니다.
sentences = re.split(r'([.!?])', output_text)  # 마침표, 느낌표, 물음표 등을 기준으로 분리

# 문장 단위로 출력
print("출력값:")
for i in range(0, len(sentences), 2):
    # 문장이 완전한 경우만 출력
    if i + 1 < len(sentences):  # 문장이 끝나는 구분자가 있을 때만
        sentence = sentences[i] + sentences[i + 1]
        # 미완성 문장이 아니면 출력
        if sentence.strip()[-1] in ['.', '!', '?']:
            print(sentence.strip())

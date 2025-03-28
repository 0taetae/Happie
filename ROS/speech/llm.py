### GPU 사용 문제 미해결 ###

### Llama 모델 초기화 및 텍스트를 입력 받아 모델로부터 답변을 생성 ###

import re
import os
import torch
from llama_cpp import Llama

# .gguf 모델 파일 경로 설정
model_path = r"C:\Users\SSAFY\Desktop\LLM\llama-3-Korean-Bllossom-8B-Q4_K_M.gguf"

# 모델 로딩 함수
def load_model():
    # 모델 경로 유무 확인
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")

    # CUDA 환경 변수 설정 (GPU 사용)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 0 사용하도록 설정

    ### GPU 상태 확인
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Switching to CPU...")

    # 디바이스 설정: GPU를 우선 사용하고, 실패하면 CPU로 자동 전환
    device = "cuda"  # 기본적으로 GPU 사용
    try:
        llama = Llama(model_path=model_path, device=device)
    except ValueError as e:
        llama = Llama(model_path=model_path, device="cpu")  # GPU 에러 발생 시 CPU로 모델 로드
    
    return llama

# LLM에 텍스트 입력 후 응답 받는 함수
def generate_response(llama, input_text):
    output = llama(input_text, max_tokens=100)

    # 출력 텍스트 처리: 문장 끝을 기준으로 분리
    if isinstance(output, dict):
        output_text = output.get('choices', [{}])[0].get('text', 'No text generated')
    else:
        output_text = output

    # 문장 끝을 기준으로 끊기
    sentences = re.split(r'([.!?])', output_text)  # 마침표, 느낌표, 물음표 등을 기준으로 분리

    # 문장 단위로 출력
    response = []
    for i in range(0, len(sentences), 2):
        # 문장이 완전한 경우만 출력
        if i + 1 < len(sentences):  # 문장이 끝나는 구분자가 있을 때만
            sentence = sentences[i] + sentences[i + 1]
            # 미완성 문장이 아니면 출력
            if sentence.strip()[-1] in ['.', '!', '?']:
                response.append(sentence.strip())
    
    return response

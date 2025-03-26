from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# 1. CUDA_VISIBLE_DEVICES 환경 변수 설정 (GPU 0만 사용)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 0만 사용하도록 설정

# 2. GPU만 사용하도록 강제
device = "cuda"  # GPU만 사용하도록 고정

# 3. GPU가 사용 가능한지 확인하고, GPU가 없으면 에러 발생
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. GPU is required!")

# 4. GPU 0에 모델 로드
try:
    print(f"Using device: {device}")
    
    # 모델 로드 (HuggingFace 모델 로딩)
    model = AutoModelForCausalLM.from_pretrained("Bllossom/llama-3.2-Korean-Bllossom-3B")
    # model = AutoModelForCausalLM.from_pretrained("teddylee777/Llama-3-Open-Ko-8B-gguf")
    model = model.to(device)  # 모델을 GPU로 이동 후 반정밀도 변환
    
    # GPU에서만 반정밀도 사용
    model = model.half()

except RuntimeError as e:
    print(f"CUDA Error: {e}")
    raise  # 에러 발생 시 종료

# 5. 입력 텍스트 토크나이즈
tokenizer = AutoTokenizer.from_pretrained("Bllossom/llama-3.2-Korean-Bllossom-3B")
# tokenizer = AutoTokenizer.from_pretrained("teddylee777/Llama-3-Open-Ko-8B-gguf")
input_text = "삼성 병원에 대해 알려줘."
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)  # 입력 텐서도 GPU로 이동
attention_mask = torch.ones(input_ids.shape).to(device)  # attention_mask도 동일

# 6. 모델로 텍스트 생성
output = model.generate(input_ids, attention_mask=attention_mask, max_length=100)

# 7. 출력 텍스트 디코딩
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)

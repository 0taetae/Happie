from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 허깅페이스의 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("MLP-KTLim/llama-3-Korean-Bllossom-8B")
model = AutoModelForCausalLM.from_pretrained("MLP-KTLim/llama-3-Korean-Bllossom-8B")

# 입력 텍스트 토크나이즈
input_text = "삼성 병원에 대해 알려줘."
input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
attention_mask = torch.ones(input_ids.shape).to("cuda")

# 모델로 텍스트 생성
output = model.generate(input_ids, attention_mask = attention_mask, max_length=300)
# 출력 테스트 디코딩
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)
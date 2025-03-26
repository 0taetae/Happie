### main.py (임시)

from llm import load_model, generate_response

# Llama 모델 로드
llama = load_model()

# 입력 텍스트 설정
input_text = "삼성병원에 대해 알려줘."

# 모델로 텍스트 생성 (응답을 문장 단위로 받기)
response = generate_response(llama, input_text)

# 출력 텍스트 출력
print("출력값:")
for sentence in response:
    print(sentence)

import sys
sys.path.append(r"C:\Users\SSAFY\Desktop\S12P21E103\ROS\speech")
from search_vector import search_hospital_info
from llama_cpp import Llama
import logging
import re

# 🔹 로깅 설정
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

# Llama 모델 로딩
model_path = r"C:\Users\SSAFY\Desktop\LLM\llama-3-Korean-Bllossom-8B.Q8_0.gguf"
logging.info(f"모델 로딩 중: {model_path}")
llama = Llama(model_path=model_path)
logging.info("모델 로딩 완료.")

# 대화 이력 저장 리스트
conversation_history = []

def generate_response(query, search_results):
    logging.debug(f"사용자 질문 수신: {query}")
    logging.debug(f"검색된 병원 정보 원본: {search_results}")

    # prompt_template = """
    #     당신은 삼성병원의 의료 시설 정보를 안내하는 AI 챗봇입니다. 사용자의 질문을 이해하고, 주어진 정보를 바탕으로 정확하고 자연스러운 답변을 제공하세요.

    #     ### 📌 이전 대화:
    #     {conversation_history}
        
    #     ### 💬 사용자 질문:
    #     {user_query}

    #     ### 🔍 검색된 병원 정보:
    #     {search_results}

    #     ---

    #     ### 🚦 응답 지침:
    #     1️⃣ **질문 분석**
    #     - 사용자가 특정 시설명(예: "응급실", "내과", "방사선 검사실")을 찾는지 확인하세요.  
    #     - 위치(층수), 서비스 내용, 운영 여부 등을 묻는지 파악하세요.  
    #     - 사용자의 **이전 질문이 현재 질문과 연결될 수 있는지 고려**하세요.  

    #     2️⃣ **정보 활용**
    #     - 제공된 검색 결과에서 **가장 적절한 정보를 찾아서 답변**하세요.  
    #     - 같은 유형의 여러 시설이 검색되었다면 **각각 정보를 제공**하세요.  
    #     - 만약 정보가 부족하거나 애매하면 **추가 질문을 유도**하세요.  

    #     3️⃣ **대화 스타일**
    #     - 질문에 대한 **짧고 명확한 답변을 먼저 제공**하세요.  
    #     - 필요하면 부가적인 정보(예: 위치 설명, 주요 서비스 등)를 추가하세요.  
    #     - **사용자가 추가 질문할 가능성이 있으면 대화를 유도하는 문장으로 마무리**하세요.  
    #         - 예: "1, 2, 3층에 있습니다. 어느 층을 찾으시나요?"  

    #     4️⃣ **정보 부족 시**
    #     - "해당 시설에 대한 정보를 찾을 수 없습니다."라고 답변하고,  
    #         사용자에게 **추가적인 질문을 요청**하세요.  
    #     - 예: "소아과 진료 시간이 궁금하시면 삼성병원 공식 홈페이지를 확인해 주세요."  
        
    #     ---

    #     이제 위 지침을 기반으로 답변을 생성하세요.

    # """
    # prompt_template = """
    #     당신은 삼성병원의 의료 시설 정보를 안내하는 AI 챗봇입니다. 특정 시설의 위치를 설명할 때는 층정보와 위지설명을 함께 합니다. 사용자의 질문을 이해하고, 주어진 정보를 바탕으로 **자연스러운 답변만**을 제공하세요. **코드는 포함하지 말고** 자연어로만 답변해주세요.

    #     ### 📌 이전 대화:
    #     {conversation_history}

    #     ### 💬 사용자 질문:
    #     {user_query}

    #     ### 🔍 검색된 병원 정보:
    #     {search_results}

    #     ---

    #     ### 답변:
    #    {user_query}에 대한 간단하고 자연스러운 답변을 작성하세요
    #     예시: "약제실은 5층으로, 엘리베이터 근처, 501호 병실 옆입니다."
    #     (※ 코드나 설명 없이 자연어로만 답변을 작성해주세요)
    # """
    # prompt_template = """
    #     당신은 삼성병원의 의료 시설 정보를 안내하는 AI 챗봇입니다. 사용자의 질문을 이해하고, 주어진 정보를 바탕으로 자연어로만을 제공하세요.

    #     이전 대화: {conversation_history}

    #     사용자 질문: {user_query}

    #     검색된 병원 정보: {search_results}

    #     답변: {user_query}에 대한 간단하고 자연어로만 작성하세요.
    
    # """
    
    prompt_template = """
당신은 삼성병원의 의료 시설 정보를 안내하는 AI 챗봇입니다. 사용자의 질문을 이해하고, 주어진 정보와 지침을 바탕으로 정확하고 자연스러운 답변을 제공하세요.

### 사용자의 질문:
{user_query}

### 검색된 병원 정보(참고할 것것):
{search_results}

---

응답 지침:
1. **질문 분석**: 사용자가 찾고 있는 정보가 무엇인지 파악하고, 이전 질문과 연결될 수 있는지 확인합니다.
2. **답변 제공**: 
   - **순수한 텍스트 형태로만 답변합니다.**
   - **HTML 태그, 마크다운(예: `**bold**`), 코드 블록을 포함하지 마세요.**
   - 예: `원무과는 2층에 있습니다.`
3. **대화 유도**: 추가적인 정보가 필요할 경우, 간단한 질문을 덧붙여서 대화를 유도합니다.  
   예: `더 궁금한 점 있으면 말씀해 주세요.`
4. **정보 부족 시**: `"해당 시설에 대한 정보를 찾을 수 없습니다."`라고 답변하고, 사용자에게 더 구체적인 질문을 유도합니다.

---

🚨 **중요: 응답에는 순수한 텍스트만 포함하고, HTML, 마크다운, 코드 블록, 특수 기호(`**`, `<br>`, `#` 등)를 사용하지 마세요.**


💡 **답변 예시** (형식을 유지할 것!)  
"원무과는 2층에 있습니다. 더 궁금한 점 있으면 말씀해 주세요."

이제 위 지침을 기반으로 **답변만 출력하세요.**
    """

    logging.debug(f"프롬프팅 : {prompt_template}")
    
    # # 🔹 search_results가 딕셔너리인지 확인
    # if isinstance(search_results, dict):
    #     metadatas = search_results.get("metadatas", [[]])
    #     logging.debug(f"메타데이터 리스트: {metadatas}")
    # else:
    #     logging.error("검색 결과의 형식이 올바르지 않습니다.")
    #     return "검색 결과를 불러오는 중 오류가 발생했습니다."

    # # 🔹 첫 번째 리스트 안의 데이터 가져오기
    # if isinstance(metadatas, list) and len(metadatas) > 0:
    #     metadatas = metadatas[0]
    #     logging.debug(f"추출된 메타데이터: {metadatas}")
    # else:
    #     metadatas = []
        
    # # 🔹 데이터 형식 확인 및 변환
    # if isinstance(metadatas, dict):  
    #     metadatas = [metadatas]  # 딕셔너리라면 리스트로 변환
    # elif not isinstance(metadatas, list):
    #     metadatas = []  # 리스트가 아니면 빈 리스트로 초기화
    
    # 🔹 `search_results`가 리스트인지 확인
    if isinstance(search_results, list) and len(search_results) > 0:
        metadatas = search_results  # 리스트이므로 그대로 사용
        logging.debug(f"검색된 메타데이터: {metadatas}")
    else:
        logging.error("검색 결과의 형식이 올바르지 않습니다.")
        return "검색 결과를 불러오는 중 오류가 발생했습니다."

    # 🔹 검색 결과 요약
    max_results = 3
    summarized_results = [
        f"{item.get('시설명', '알 수 없음')}: {item.get('층정보', '정보 없음')}, {item.get('위치설명', '정보 없음')}"
        for item in metadatas[:max_results]
    ]
    search_results_str = "\n".join(summarized_results)

    # 🔹 이전 대화 내역을 문자열로 변환
    conversation_history_str = "\n".join([f"👤 사용자: {item['user']}\n 챗봇: {item['response']}" for item in conversation_history])
    
    # 🔹 프롬프트 포맷팅
    prompt = prompt_template.format(
        user_query=query,
        search_results=search_results_str,
        conversation_history=conversation_history_str
    )

    logging.debug(f"생성된 프롬프트:\n{prompt}")

    # 🔹 Llama 모델 실행
    logging.info("Llama 모델을 사용해 응답 생성 중...")
    response = llama(prompt, max_tokens=50, temperature=0.2)
    # response = llama(prompt, max_tokens=200)
    logging.info(f"응답: {response}")
    logging.info("응답 생성 완료.")

    # 🔹 응답이 텍스트 형식으로 반환되도록 처리
    # response_text = response.get('choices', [{}])[0].get('text', 'No response generated.').strip()
    
    # logging.debug(f"원본 응답: {response_text}")

    # # 🔹 응답 텍스트 정제
    # if "질문에 대한 답변:" in response_text:
    #     split_text = response_text.split("질문에 대한 답변:", 1)
    #     response_text = split_text[1].strip() if len(split_text) > 1 else response_text

    # logging.debug(f"최종 응답: {response_text}")
    # return response_text
    
        # 🔹 응답이 텍스트 형식으로 반환되도록 처리
     # 🔹 응답에서 텍스트만 추출
    response_text = response.get('choices', [{}])[0].get('text', 'No response generated.').strip()
    
    logging.debug(f"원본 응답: {response_text}")

    # 🔹 "챗봇:"으로 시작하는 부분 제거
    response_text = response_text.replace("챗봇: ", "", 1).strip()

    # 🔹 "(이 응답은 예시입니다..." 이후 내용 제거
   # 🔹 줄바꿈("\n")을 기준으로 마지막 줄만 가져오기
    last_line = response_text.strip().split("\n")[0]

    logging.debug(f"최종 응답: {last_line}")
    return last_line


# 🔹 대화형 챗봇 실행 함수
def chat():
    print("챗봇을 시작합니다. 종료하려면 'exit'을 입력하세요.")

    while True:
        user_input = input("사용자: ")
        logging.info(f"사용자 입력: {user_input}")

        if user_input.lower() == 'exit':
            logging.info("챗봇 종료")
            print("챗봇을 종료합니다.")
            break

        search_results = search_hospital_info(user_input)
        logging.debug(f"검색 결과 반환: {search_results}")
        
        # 한 줄로 변환하고 리스트
        converted_data = [
            {
                '시설명': item['facility_name'],
                '층정보': item['floor_info'],
                '위치설명': item['service_description'],  # 'service_description'을 '위치설명'으로 변경
                '위치': item['location'],  # 추가적으로 'location'도 한국어로 변경 가능
                '점수': item['score']  # 필요하면 점수도 포함
            }
            for item in search_results
        ]

        if not search_results:
            logging.warning("검색 결과 없음")
            print("❌ 관련된 정보를 찾을 수 없습니다.")
            continue


        # response = generate_response(user_input, search_results)
        response = generate_response(user_input, converted_data)

        print(f"챗봇: {response} 이상 끝!")


if __name__ == "__main__":
    chat()

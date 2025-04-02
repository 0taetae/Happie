import sys
sys.path.append(r"C:\Users\SSAFY\Desktop\S12P21E103\ROS\speech")
from search_vector import search_hospital_info
from llama_cpp import Llama
import logging
# from memory_manager import ConversationMemoryManager  # memory_manager.py에서 정의한 메모리 관리 클래스를 임포트


# 🔹 로깅 설정
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

# Llama 모델 로딩
model_path = r"C:\Users\SSAFY\Desktop\LLM\llama-3-Korean-Bllossom-8B.Q8_0.gguf"
logging.info(f"모델 로딩 중: {model_path}")
llama = Llama(model_path=model_path)
logging.info("모델 로딩 완료.")

### 메모리 매니저 인스턴스 생성
# memory_manager = ConversationMemoryManager()

# 대화형 챗봇 생성 함수
def generate_response(query, search_results):
    logging.debug(f"사용자 질문 수신: {query}")
    logging.debug(f"검색된 병원 정보 원본: {search_results}")
    
    # 검색된 병원 정보를 텍스트 형식으로 요약
    summarized_results = [
        f"{item.get('facility_name', '알 수 없음')}: {item.get('floor_info', '정보 없음')} | {item.get('location', '정보 없음')} | {item.get('service_description', '정보 없음')}"
        for item in search_results[:3]  # 최대 3개 결과만 요약
    ]
    search_results_str = "\n".join(summarized_results)
    
    ### 이전 대화 이력 가져오기
    # previous_conversations = memory_manager.get_previous_conversations()
    
    # 프롬프트 템플릿 구성
    prompt_template = """
    너는 삼성병원의 의료 시설 정보를 안내하는 AI야. 사용자의 질문을 이해하고, 검색된 병원 정보를 바탕으로 정확하고 자연스러운 답변을 제공해.
    
    ### 질문:
    {user_query}  

    ### 검색된 병원 정보:
    {search_results}  

    ---

    응답 규칙:  
    1. **질문을 분석**해서 사용자가 원하는 정보를 찾아.  
    2. **검색된 정보가 있으면**, 해당 내용을 정확한 높임말로 전달해.  
    - 예: "응급실은 1층에 있습니다."  
    3. **검색된 정보가 없으면**, `"해당 정보를 찾을 수 없습니다."`라고 답하고, 추가 질문을 유도해.  
    4. **사용자가 잘못 알고 있을 경우**, 올바른 정보를 제공해.  
    - 예: "응급실은 2층이 아니라 1층에 있습니다."  
    5. **응답은 항상 높임말로 작성**해.  

    답변만 출력해.  
    """


    # 프롬프트 포맷팅
    prompt = prompt_template.format(
        # previous_conversations=previous_conversations,
        user_query=query,
        search_results=search_results_str
    )

    logging.debug(f"프롬프트 생성:\n{prompt}")

    # 🔹 Llama 모델 실행하여 응답 생성
    logging.info("Llama 모델을 사용해 응답 생성 중...")
    response = llama(prompt, max_tokens=50, temperature=0.2)
    logging.info(f"응답 생성 완료.")

    # 🔹 응답에서 텍스트만 추출하고, 불필요한 부분 제거
    response_text = response.get('choices', [{}])[0].get('text', 'No response generated.').strip()
    response_text = response_text.replace("챗봇: ", "", 1).strip()
    
    # 🔹 후처리: 뒤에서부터 가장 가까운 마침표까지 유지
    last_period_index = response_text.rfind(".")
    if last_period_index != -1:  # 마침표가 있으면 해당 부분까지만 남김
        response_text = response_text[: last_period_index + 1]

    logging.debug(f"최종 응답: {response_text}")
    return response_text


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

        # 검색 결과를 받아오기
        search_results = search_hospital_info(user_input)
        logging.debug(f"검색 결과 반환: {search_results}")
        
        # 검색 결과가 없다면 경고 메시지
        if not search_results:
            logging.warning("검색 결과 없음")
            print("❌ 관련된 정보를 찾을 수 없습니다.")
            continue

        # 검색 결과를 처리하여 응답 생성
        generate_response(user_input, search_results)
        # response = generate_response(user_input, search_results)
        # print(f"챗봇: {response} 이상 끝!")


if __name__ == "__main__":
    chat()

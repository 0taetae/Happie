import sys
sys.path.append(r"C:\Users\SSAFY\Desktop\S12P21E103\ROS\speech")
from search_chromadb import search_hospital_info
import logging
import openai
from dotenv import load_dotenv
import os
from hospital_google_search import google_search
from tavily_search import tavily_search, optimize_query
import re

# load .env
load_dotenv()

API_KEY = os.environ.get('API_KEY')

# 🔹 로깅 설정
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

# OpenAI API 키 설정
client = openai.OpenAI(api_key=API_KEY)


# 🔹 대화 히스토리 저장용 리스트
history = []


# 대화형 챗봇 생성 함수
def generate_response(query, search_results, external_search):
    logging.debug(f"사용자 질문 수신: {query}")
    logging.debug(f"검색된 병원 정보 원본: {search_results}")
    
    # 검색된 병원 정보를 텍스트 형식으로 요약
    summarized_results = [
        f"{item.get('facility_name', '알 수 없음')}: {item.get('floor_info', '정보 없음')} | {item.get('location', '정보 없음')} | {item.get('service_description', '정보 없음')}"
        for item in search_results[:5]
    ]
    search_results_str = "\n".join(summarized_results)

    messages=[
            {
                "role": "system",
                "content": f"""
                    너는 삼성병원의 의료 시설 정보를 안내하고 일상대화도 가능한 AI챗봇이이야.
                    사용자의 질문을 이해하고, 검색된 병원 정보를 바탕으로 정확하고 자연스러운 답변을 제공해.

                    현재 제공할 수 있는 병원 정보:
                    {search_results_str if search_results else "현재 제공할 수 있는 병원 정보가 없습니다."}

                    외부 검색 정보: 
                    {external_search if external_search else "현재 제공할 수 있는 외부 검색 정보가 없습니다."}

                    응답 규칙:
                    1. **질문을 분석**해서 사용자가 원하는 정보를 찾아.
                    2. **검색된 정보가 있으면**, 해당 내용을 정확한 높임말로 전달해.
                       - 예: "응급실은 1층에 있습니다."
                    3. **검색된 정보가 없으면**, `"해당 정보를 찾을 수 없습니다."`라고 답하거나 너가 답할 수 있는거면 답해주거나 추가 질문을 유도해.
                    4. **사용자가 잘못 알고 있을 경우**, 올바른 정보를 제공해.
                       - 예: "응급실은 2층이 아니라 1층에 있습니다."
                    5. **응답은 항상 높임말로 작성**해.

                    답변만 출력해.
                """
            },
            # {
            #     "role": "user",
            #     "content": query
            # }
        ]

    # 🔹 기존 대화 히스토리 추가
    messages.extend(history)

    # 🔹 현재 질문 추가
    messages.append({"role": "user", "content": query})


    logging.info(f"메시지 출력 : {messages}")

    # 🔹 OpenAI GPT 모델 호출
    logging.info("GPT 모델을 사용해 응답 생성 중...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 또는 "gpt-3.5-turbo"
        messages=messages,
        max_tokens=150,
        temperature=0.7
    )

    print(f"응답: {response}")
    response_text = response.choices[0].message.content

    # ✅ 새로운 대화 내역 저장
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": response_text})


    response_text = response.choices[0].message.content

    logging.debug(f"최종 응답: {response_text}")
    return response_text

# history 초기화 함수
def clear_history():
    global history
    history = []
    print("대화 기록이 초기화되었습니다.")


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
        # if not search_results:
        #     logging.warning("검색 결과 없음")
        #     print("❌ 관련된 정보를 찾을 수 없습니다.")
        #     continue

        #외부 검색
        external_search= []
    
        # 최적화된 검색어 생ㅇ성
        optimized_query = optimize_query(user_input)
        logging.info("최적화된 검색어: ", optimized_query)

        # Google 검색 결과 추가
        google_results = google_search("site:samsunghospital.com " , optimized_query)
        if google_results:
            logging.info("Google 검색 결과 추가")
            external_search.extend(google_results)  # 기존 검색 결과에 추가
            print("🔎 Google 검색 결과를 추가했습니다.")
        
        # tavily_search_answer = tavily_search(optimized_query)
        # if tavily_search_answer:
        #     logging.info("tavily 검색 결과 추가")
        #     external_search.extend(tavily_search_answer)

        # 검색 결과를 처리하여 응답 생성
        print(search_results)
        print("외부 검색 결과: " , external_search)
        response = generate_response(user_input, search_results, external_search)




if __name__ == "__main__":
    chat()
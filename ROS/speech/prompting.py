import sys
import os
import logging
import openai
from dotenv import load_dotenv
from search_chromadb import search_hospital_info
from weather_api import get_weather_with_context
from hospital_google_search import google_search
from tavily_search import tavily_search, optimize_query
import re

load_dotenv()
API_KEY = os.environ.get('API_KEY')

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

# OpenAI API 키 설정
client = openai.OpenAI(api_key=API_KEY)

# 대화 히스토리 저장용 리스트
history = []

# 응답 형태 후처리 : 응답의 마지막 마침표까지의 내용만 반환
def trim_to_complete_sentence(response_text):
    last_period_index = response_text.rfind(".")

    if last_period_index != -1:
        return response_text[: last_period_index + 1]  # 마침표 포함한 부분까지만 반환
    return response_text  # 마침표가 없으면 원문 그대로 반환

# 대화형 챗봇 생성 함수
def generate_response(query, search_results, external_search):
    logging.debug(f"사용자 질문 수신: {query}")
    logging.debug(f"검색된 병원 정보 원본: {search_results}")

   # 날씨 질문 감지
    if "날씨" in query or "기온" in query:
        weather_info = get_weather_with_context("Seoul")  # 서울로 고정
        return f"서울의 현재 날씨는 {weather_info} 입니다."

    # 병원 정보 응답
    summarized_results = [
        f"{item.get('facility_name', '알 수 없음')}: {item.get('floor_info', '정보 없음')} | {item.get('location', '정보 없음')} | {item.get('service_description', '정보 없음')}"
        for item in search_results[:5]
    ]
    search_results_str = "\n".join(summarized_results)

    messages=[
        {
            "role": "system",
            "content": f"""
                너는 삼성병원의 의료 시설 정보를 안내하는 AI야. 이름은 '하피'라고 해.
                사용자의 질문을 이해하고, 검색된 병원 정보를 바탕으로 정확하고 자연스러운 답변을 제공해.
                또한 사용자의 요청에 따라 다양한 응답을 할 수 있어.

                응답 규칙:
                1. **사용자의 질문을 분석**해서 원하는 정보를 찾아.
                2. **검색된 병원 정보가 있으면**, 해당 내용을 정확한 높임말로 전달해.
                3. **검색된 병원 정보와 외부 검색 정보가 모두 없으면**, "해당 정보를 찾을 수 없습니다."라고 답하고, 추가 질문을 유도해.
                4. **외부 검색 정보가 있으면**, 해당 내용을 바탕으로 유용한 정보를 제공하고 추가 질문을 유도해.
                4. **사용자가 날씨를 물어볼 경우**, '서울'의 날씨 정보를 제공해.
                5. **응답은 항상 높임말로 작성**해.

                현재 제공할 수 있는 병원 정보:
                {search_results_str if search_results else "현재 제공할 수 있는 병원 정보가 없습니다."}
                외부 검색 정보:
                {external_search if external_search else "현재 제공할 수 있는 외부 검색 정보가 없습니다."}

                완성된 문장으로 답변을 하자.
            """
        }
    ]

    messages.extend(history)
    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150,
        temperature=0.7
    )

    # OepnAI API로 생성된 응답 가져오기
    response_text = response.choices[0].message.content

    # 결과 후처리
    response_text = trim_to_complete_sentence(response_text)

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": response_text})

    return response_text

# history 초기화 함수
def clear_history():
    global history
    history = []
    print("대화 기록이 초기화되었습니다.")

# 대화형 챗봇 실행 함수
def chat():
    print("챗봇을 시작합니다. 종료하려면 'exit'을 입력하세요.")

    while True:
        user_input = input("사용자: ")

        if user_input.lower() == 'exit':
            print("챗봇을 종료합니다.")
            break



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

        response = generate_response(user_input, search_hospital_info(user_input), external_search)
        print(response)
if __name__ == "__main__":
    chat()

import os
from dotenv import load_dotenv  # 🔹 환경 변수 로드
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from weather_api import weather_tool  # ✅ 우리가 만든 날씨 툴 가져오기

# 🔹 .env 파일에서 환경 변수 불러오기
load_dotenv()

# 🔹 OpenAI 및 OpenWeatherMap API 키 불러오기
OPENAI_API_KEY = os.getenv("API_KEY")  # ✅ OpenAI GPT 키
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")  # ✅ OpenWeatherMap 키

# 🔹 LangChain의 OpenAI 모델 설정
llm = ChatOpenAI(model_name="gpt-4", temperature=0.5, openai_api_key=OPENAI_API_KEY)

# 🔹 LangChain Agent 설정 (날씨 API 툴 추가)
agent = initialize_agent(
    tools=[weather_tool],  # ✅ OpenWeatherMap을 연결
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # GPT가 자동으로 적절한 도구를 선택하도록 설정
    verbose=True
)

# ✅ 테스트 실행
if __name__ == "__main__":
    print(agent.run("오늘 서울 날씨 어때?"))  # LangChain이 날씨 API를 자동 호출

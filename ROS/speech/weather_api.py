import os
from dotenv import load_dotenv
import requests
from langchain.tools import Tool

# 🔹 .env 파일에서 환경 변수 불러오기
load_dotenv()

# 🔹 OpenWeatherMap API 키 불러오기
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_weather(city: str) -> str:
    """도시명을 받아서 OpenWeatherMap API에서 날씨 정보를 가져오는 함수"""
    if not OPENWEATHER_API_KEY:
        return "🔴 API 키가 설정되지 않았습니다."

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&lang=kr&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        weather_description = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        return f"🌤 {city}의 현재 날씨: {weather_description}, 기온 {temp}°C"
    else:
        return f"⚠️ '{city}'의 날씨 정보를 가져올 수 없습니다."

# 🔹 LangChain Tool로 등록
weather_tool = Tool(
    name="WeatherAPI",
    func=get_weather,  # ✅ 함수 참조 방식으로 등록
    description="도시 이름을 입력하면 현재 날씨를 알려줍니다."
)

# ✅ 테스트 실행
if __name__ == "__main__":
    print(get_weather("Seoul"))  # 🔥 서울 날씨 테스트

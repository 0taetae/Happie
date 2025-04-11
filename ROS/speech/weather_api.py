import os
import requests
from dotenv import load_dotenv

load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_weather_with_context(city: str = "Seoul") -> str:
    """서울의 날씨 정보를 OpenWeatherMap API에서 가져오는 함수"""
    if not OPENWEATHER_API_KEY:
        return "🔴 API 키가 설정되지 않았습니다."

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&lang=kr&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        weather_description = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        return f"{weather_description}, 기온은 {temp}°C"
    else:
        return "⚠️ 서울의 날씨 정보를 가져올 수 없습니다."

if __name__ == "__main__":
    print(get_weather_with_context())  # 서울 날씨 테스트

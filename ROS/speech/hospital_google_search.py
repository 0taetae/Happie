import requests, os, re
from dotenv import load_dotenv


# .env 파일 로드
load_dotenv()

# ✅ 환경 변수에서 API 키와 검색 엔진 ID 가져오기
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Google API 키
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")  # Custom Search Engine ID


def google_search(query, num_results=3):
    query = extract_keywords(query)
    """Google Custom Search API를 사용해 검색 결과 반환"""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": num_results,
    }

    response = requests.get(url, params=params)
    data = response.json()
    # print(data)

    # 결과 처리
    search_results = []
    for item in data.get("items", []):
        search_results.append({
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet"),
        })
    
    return search_results

def extract_keywords(query):
    """사용자 입력에서 핵심 키워드만 추출"""
    stopwords = {"은", "는", "이", "가", "에", "대해", "알려줘", "정보", "어디"}  # 불필요한 단어 리스트
    words = re.findall(r"\b\w+\b", query)  # 단어 단위로 분리
    keywords = [word for word in words if word not in stopwords]
    return " ".join(keywords)

# print(google_search("site:samsunghospital.com 소아과 의료진 목록"))
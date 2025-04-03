import os
from dotenv import load_dotenv
from tavily import TavilyClient

# .env 파일 로드
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # Tavily API 키

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY가 설정되지 않았습니다! .env 파일을 확인하세요.")

def tavily_search(query, num_results=1):
    """Tavily API를 사용한 웹 검색 (서울삼성병원 관련 정보만 검색)"""
    client = TavilyClient(api_key=TAVILY_API_KEY)

    # 항상 "서울삼성병원"을 포함하여 검색
    full_query = f"서울삼성병원 {query}"

    # 검색 실행
    response = client.search(
        query=full_query,  
        search_depth="advanced",
        max_results=num_results
    )

    # 🔍 API 응답 구조 확인
    # print("📡 API 응답 데이터:", response)

    # 결과 정리
    if not isinstance(response, dict):
        return "❌ 잘못된 API 응답 형식입니다."

    results = response.get("results", [])
    if not results:
        return "❌ 관련된 정보를 찾을 수 없습니다."

    formatted_results = []
    
    for item in results:
        title = item.get("title", "제목 없음")
        url = item.get("link") or item.get("url", "URL 없음")  # `link`와 `url` 중 존재하는 것 사용
        snippet = item.get("content", "내용 없음")
        
        # formatted_results.append(f"📌 **{title}**\n🔗 {url}\n📝 {snippet[:400]}...")  # 내용 일부만 표시
        formatted_results.append(snippet[:400])

    return formatted_results

# ✅ 테스트
# result = tavily_search("소아과 의료진", num_results=2)
# print(result)

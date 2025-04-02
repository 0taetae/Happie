### 벡터 데이터 기반 검색 ###
# 사용자 입력 정보(텍스트)를 벡터화 > 시설명 정확 일치 검색 + 의미적 유사도 검색

import chromadb
import re
from langchain.embeddings import OpenAIEmbeddings
import openai
from dotenv import load_dotenv
import os

# .env 연결
load_dotenv()

#OPENAPI 변수 : OpenAI API 키 환경 변수에서 가져오기
API_KEY = os.environ.get('API_KEY')

# OpenAI API 키 설정
openai.api_key = API_KEY

# ChromaDB 연결
def connect_chromadb():
    try:
        print("ChromaDB 연결 중...")
        client = chromadb.HttpClient(host='j12e103.p.ssafy.io', port=8000)
        collection = client.get_collection("hospital_info")
        return collection
    except Exception as e:
        print(f"ChromaDB 연결 오류: {e}")
        return None


# 사용자 질문(텍스트) 벡터화
def vectorize_query(query):
    print("사용자 질문 벡터화 중...")
    embeddings_model = OpenAIEmbeddings(openai_api_key=API_KEY, model="text-embedding-ada-002")
    return embeddings_model.embed_query(query)


# 병원 정보 검색
def search_hospital_info(query):
    collection = connect_chromadb()
    if not collection:
        return []

    # 1️⃣ 사용자 질문 정제
    query_cleaned = clean_query(query)
    
    # 2️⃣ 시설명 정확 일치 검색 (여러 개 가능)
    facility_names = extract_facility_names(query_cleaned, collection)
    exact_matches = []

    if facility_names:
        results = collection.get(where={"시설명": {"$in": facility_names}})
        if results['documents']:
            exact_matches.extend(format_results(results))

    # 3️⃣ 의미적 유사도 검색 (시설명이 없거나, 보완 검색이 필요한 경우)
    query_embedding = vectorize_query(query_cleaned)
    search_results = collection.query(query_embeddings=[query_embedding], n_results=3)
    semantic_results = format_results(search_results)

    # 4️⃣ 시설명 검색 결과가 있다면 최우선적으로 포함하고, 부족한 경우 유사도 검색 결과 포함
    final_results = exact_matches + semantic_results
    unique_results = {res['facility_name']: res for res in final_results}  # 중복 제거

    return list(unique_results.values())[:3]  # 최종 3개까지만 반환


### STT 결과 처리 : 불필요한 문장 요소 제거
def clean_query(query):
    query = query.strip()  # 앞뒤 공백 제거
    query = re.sub(r'[^가-힣0-9\s]', '', query)  # 한글, 숫자, 공백 제외한 문자 제거
    query = re.sub(r'\s+', ' ', query)  # 연속된 공백 하나로 변환

    # 조사 및 불필요한 단어 제거 (예: "은", "는", "이", "가", "을", "를" 등)
    stopwords = ["은", "는", "이", "가", "을", "를", "에", "서", "어디", "어떻게", "있어", "가", "니", "요"]
    words = query.split()
    words = [word for word in words if word not in stopwords]
    
    return ' '.join(words)  # 공백 기준으로 다시 합침


### 질문에서 시설명 추출 : 띄어쓰기 오류 보정 + 층 정보 무시
def extract_facility_names(query, collection):
    """질문에서 층 정보(숫자+층)를 제거하고 시설명 추출"""
    query = re.sub(r'\d+층', '', query)  # "1층", "2층" 등 제거
    query = query.strip()
    
    all_facilities = collection.get(include=["metadatas"])  
    raw_metadatas = all_facilities.get("metadatas", [])  

    facility_names = set()
    for meta_list in raw_metadatas:  
        if isinstance(meta_list, list):
            for meta in meta_list:
                if isinstance(meta, dict):  
                    facility_names.add(meta.get("시설명", ""))

    # 띄어쓰기 제거 후 검색 (여러 개 반환)
    query_words = query.replace(" ", "")
    matched_facilities = [name for name in facility_names if name.replace(" ", "") in query_words]

    return matched_facilities if matched_facilities else []  # 여러 개 반환 가능


# 검색 결과를 딕셔너리 리스트로 변환
def format_results(results):
    formatted = []
    
    # ChromaDB 결과에서 문서 및 메타데이터 가져오기
    for metadata_list in results.get("metadatas", []):
        if isinstance(metadata_list, list):
            for metadata in metadata_list:
                formatted.append({
                    'facility_name': metadata.get('시설명', '정보 없음'),
                    'floor_info': metadata.get('층정보', '정보 없음'),
                    'service_description': metadata.get('서비스설명', '정보 없음'),
                    'location': metadata.get('위치설명', '정보 없음')
                })
        else:
            formatted.append({
                'facility_name': metadata_list.get('시설명', '정보 없음'),
                'floor_info': metadata_list.get('층정보', '정보 없음'),
                'service_description': metadata_list.get('서비스설명', '정보 없음'),
                'location': metadata_list.get('위치설명', '정보 없음')
            })
    
    return formatted


if __name__ == "__main__":
    query = input("검색할 내용을 입력하세요: ")
    results = search_hospital_info(query)
    for result in results:
        print(result)

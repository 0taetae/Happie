import chromadb
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

def search_hospital_info(query):
    print("ChromaDB 연결 중...")
    client = chromadb.HttpClient(host='j12e103.p.ssafy.io', port=8000)
    collection = client.get_collection("hospital_info")
    
    # 임베딩 모델 로드 : 질문을 벡터화
    print("질문 벡터화 중...")
    # embeddings_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
    embeddings_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
    query_embedding = embeddings_model.embed_query(query)
    
    # 유사도 검색 : 최대 5개 산출
    search_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    # 유사도 검색 결과 출력
    print(search_results)
    
    # 유사도 임계값 설정
    threshold = dynamic_threshold(search_results)
    
    # 결과 필터링 및 출력
    results = []
    for doc, metadata, score in zip(search_results["documents"], search_results["metadatas"], search_results["distances"]):
        if isinstance(score, list):  # score가 리스트이면 첫 번째 값만 사용
            score = score[0] if score else 0.0  # 리스트가 비어 있으면 0.0

        # metadata가 리스트인 경우 첫 번째 요소를 가져오도록 수정
        metadata = metadata[0] if isinstance(metadata, list) and metadata else {}

        result = {
            'facility_name': metadata.get('시설명', '정보 없음'),
            'floor_info': metadata.get('층정보', '정보 없음'),
            'service_description': metadata.get('서비스설명', '정보 없음'),
            'location' : metadata.get('위치설명', '정보 없음'),
            'score': score
        }
        results.append(result)
    
    # 유사도 기준으로 내림차순 정렬
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    return results

# 유사도 임계값을 동적으로 설정하는 함수
### 현재 코드에서 임계값만 계산하고 사용하지 않는 중
def dynamic_threshold(search_results):
    scores = search_results.get("distances", [])

    # 예외 처리: 만약 scores가 비어 있거나 None이면 기본값 반환
    if not scores or not isinstance(scores, list):
        return 0.0

    # scores가 2차원 리스트일 경우 평탄화
    if isinstance(scores[0], list):
        scores = [item for sublist in scores for item in sublist]

    if not scores:  # 평탄화 후에도 비어 있다면 0.0 반환
        return 0.0

    max_score = max(scores)
    min_score = min(scores)
    
    return (max_score + min_score) / 2

if __name__ == "__main__":
    query = input("검색할 내용을 입력하세요: ")
    
    results = search_hospital_info(query)

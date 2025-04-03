### chromadb 벡터 데이터 저장 ###

import chromadb
from vectorization import vectorize_data
from excel_load import load_excel

# 데이터를 수신했을 때 저장
def store_in_chromadb():
    # ChromaDB 클라이언트 설정
    print("ChromaDB 클라이언트 연결 중...")
    client = chromadb.HttpClient(host='j12e103.p.ssafy.io', port=8000)
    
    # 기존 컬렉션 삭제 후 재생성
    client.delete_collection("hospital_info")
    print("기존 hospital_info 컬렉션 삭제 완료.")
    collection = client.create_collection("hospital_info")
    print("새로운 hospital_info 컬렉션 생성 완료.")
    
    # 엑셀 데이터 로드 및 벡터화
    df = load_excel()
    embeddings, metadatas = vectorize_data(df)
    
    # ChromaDB에 데이터 추가
    for i, (embedding, metadata) in enumerate(zip(embeddings, metadatas)):
        collection.add(
            ids=[str(i)],  # 각 데이터의 고유 ID (문자열로 변환)
            embeddings=[embedding],  # 벡터화된 데이터
            metadatas=[metadata],  # 메타데이터 (시설명, 위치설명, 서비스설명, 층정보 포함)
            documents=[df["combined_text"].iloc[i]] # 문서 자체
        )
    
    print("병원 데이터가 ChromaDB에 저장되었습니다!")

if __name__ == "__main__":
    store_in_chromadb()

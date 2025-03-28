### chromadb 벡터 데이터 저장 ###

import chromadb
import json

# Chromadb 클라이언트 설정
client = chromadb.HttpClient(host='j12e103.p.ssafy.io', port=8000)

# 데이터를 수신했을 때 이를 저장하는 함수
def store_in_chromadb(message):
    client.delete_collection("hospital_info")  # 기존 컬렉션 삭제
    print("기존 hospital_info 컬렉션이 삭제되었습니다.")
    collection = client.create_collection("hospital_info")  # 컬렉션 이름 지정
    print("새로운 hospital_info 컬렉션이 생성되었습니다.")

    # 메시지에서 필요한 데이터를 가져옵니다.
    facility_names = message["시설명"]
    location_descriptions = message["위치설명"]
    service_descriptions = message["서비스설명"]
    floor_info = message["층정보"]
    
    facility_embeddings = message["facility_embeddings"]
    location_embeddings = message["location_embeddings"]
    service_embeddings = message["service_embeddings"]
    floor_embeddings = message["floor_embeddings"]
    
    # 저장할 데이터 형식 설정
    documents = facility_names  # 예시로 시설명을 documents로 설정
    metadatas = {
        "위치설명": location_descriptions,
        "서비스설명": service_descriptions,
        "층정보": floor_info
    }
    
    embeddings = {
        "facility_embeddings": facility_embeddings.tolist(),
        "location_embeddings": location_embeddings.tolist(),
        "service_embeddings": service_embeddings.tolist(),
        "floor_embeddings": floor_embeddings.tolist()
    }
    
    # 데이터 추가 (문서, 메타데이터, 벡터 저장)
    collection.add(
        documents=documents, 
        metadatas=metadatas, 
        embeddings=embeddings
    )
    
    print("데이터가 chromadb에 저장되었습니다.")


from fastapi import FastAPI
import chromadb
from chromadb.config import Settings

import logging

logging.basicConfig(level=logging.INFO)

# FastAPI 인스턴스 생성
app = FastAPI()

# ChromaDB 클라이언트 설정 (EC2에 있는 ChromaDB 사용)
client = chromadb.HttpClient(host='http://j12e103.p.ssafy.io:8000')

# 컬렉션 목록 확인
collections = client.list_collections()
for col_name in collections:
    print(f"컬렉션 이름: {col_name}")

collections = client.list_collections()
for col_name in collections:
    collection = client.get_collection(col_name)  # 컬렉션 객체 가져오기
    print(f"컬렉션 이름: {collection.name}")
    # 여기서 collection 객체로 추가 작업 가능

collection = client.get_collection("hospital_info")
print(collection.peek())

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI with ChromaDB!"}

# 컬렉션 확인 엔드포인트
@app.get("/collections")
def check_collections():
    collections = client.list_collections()  # 모든 컬렉션을 가져옴
    return {"collections": collections}

# 특정 컬렉션을 조회하는 엔드포인트
@app.get("/collections/hospital_info")
def get_collect():
    logging.info("### 엔드포인트 시작 ###")
    try:
        # logging.info(f"요청된 컬렉션 이름: {collection_name}")
        collection = client.get_collection("hospital_info")
        logging.info("컬렉션 가져오기 성공")
        return {
            "콜렉션" : collection.peek()
        }
    except Exception as e:
        logging.error(f"에러 발생: {str(e)}")
        return {"error": f"컬렉션 조회 중 오류 발생: {str(e)}"}



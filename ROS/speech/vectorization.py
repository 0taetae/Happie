### 데이터 벡터화 (LangChain 활용) ###

from langchain.embeddings import OpenAIEmbeddings
import openai
from excel_load import load_excel
import pandas as pd
from dotenv import load_dotenv
import os

# .env 연결
load_dotenv()

#OPENAPI 변수 : OpenAI API 키 환경 변수에서 가져오기
API_KEY = os.environ.get('API_KEY')

# OpenAI API 키 설정
openai.api_key = API_KEY

def vectorize_data(df):
    print("OpenAI 임베딩 모델 로드 중...")
    # OpenAI 임베딩 모델을 사용
    embeddings_model = OpenAIEmbeddings(openai_api_key=API_KEY, model="text-embedding-ada-002")
    # embeddings_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
    
    # 시설명 + 위치설명 + 서비스설명 + 층정보를를 하나로 합쳐서 벡터화
    df["combined_text"] = df[["시설명", "위치설명", "서비스설명", "층정보"]].fillna('').agg(' '.join, axis=1)
    
    print("병원 데이터 벡터화 중...")
    text_embeddings = embeddings_model.embed_documents(df["combined_text"].tolist())
    
    # 층정보는 metadata에도 저장
    metadatas = df[["시설명", "위치설명", "서비스설명", "층정보"]].to_dict(orient="records")
    
    return text_embeddings, metadatas

if __name__ == "__main__":
    df = load_excel()
    embeddings, metadatas = vectorize_data(df)
    print("벡터화 완료!")
### 데이터 벡터화 (LangChain 활용) ###

from langchain_community.embeddings import HuggingFaceEmbeddings
from excel_load import load_excel
import pandas as pd

def vectorize_data(df):
    # Sentence-BERT 모델
    ### LangChain의 HuggingFaceEmbeddings를 활용해서 벡터화 코드 간소화
    print("한국어 임베딩 모델 로드 중...")
    embeddings_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
    
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
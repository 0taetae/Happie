### 데이터 벡터화 ###

from langchain_community.embeddings import HuggingFaceEmbeddings
from excel_load import load_excel  # excel_load.py의 load_excel 함수
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.preprocessing import LabelEncoder

# 텍스트 벡터화 함수
def vectorize_data(df):
    # 한국어 모델을 사용하여 임베딩
    print("한국어 모델 로드 중...")
    
    # 시설명 벡터화를 위한 BERT 모델
    facility_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    facility_model = AutoModel.from_pretrained("bert-base-multilingual-cased")
    
    # 위치 설명, 서비스 설명 벡터화를 위한 Sentence-BERT 모델
    location_service_tokenizer = AutoTokenizer.from_pretrained("jhgan/ko-sbert-sts")
    location_service_model = AutoModel.from_pretrained("jhgan/ko-sbert-sts")
    
    # '시설명', '위치설명', '서비스설명' 컬럼 벡터화
    def vectorize_bert(texts, tokenizer, model):
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # 문장의 평균 임베딩 벡터
        return embeddings

    def vectorize_sentence_bert(texts, tokenizer, model):
        # 입력값이 리스트가 아닌 경우, 리스트로 변환
        if not isinstance(texts, list):
            texts = [texts]
            
        # texts가 리스트인지 확인
        if not isinstance(texts, list):
            raise ValueError(f"Expected list of texts, got {type(texts)}")
        
        # 텍스트들을 토큰화하고 모델을 통해 벡터화
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.pooler_output  # Sentence-BERT의 pooler_output
        return embeddings
    
    # 시설명 벡터화
    print("시설명 벡터화 중...")
    facility_names = df['시설명'].fillna('')  # 결측값 처리
    facility_embeddings = vectorize_bert(facility_names.tolist(), facility_tokenizer, facility_model)
    
    # 위치설명 벡터화
    print("위치설명 벡터화 중...")
    location_descriptions = df['위치설명'].fillna('')  # 결측값 처리
    location_embeddings = vectorize_sentence_bert(location_descriptions.tolist(), location_service_tokenizer, location_service_model)

    # 서비스설명 벡터화
    print("서비스설명 벡터화 중...")
    service_descriptions = df['서비스설명'].fillna('')  # 결측값 처리
    service_embeddings = vectorize_sentence_bert(service_descriptions.tolist(), location_service_tokenizer, location_service_model)

    # 층정보 벡터화 (Label Encoding으로 수정)
    print("층정보 벡터화 중...")
    floor_info = df['층정보'].fillna('')  # 결측값 처리
    floor_info = floor_info.astype(str)  # 문자열로 처리
    encoder = LabelEncoder()
    floor_embeddings = encoder.fit_transform(floor_info)  # 층 정보를 0, 1, 2와 같은 숫자로 변환
    
    return facility_embeddings, location_embeddings, service_embeddings, floor_embeddings

if __name__ == "__main__":
    # 엑셀 파일 로드
    df = load_excel()
    
    # 벡터화 함수 실행
    facility_embeddings, location_embeddings, service_embeddings, floor_embeddings = vectorize_data(df)
    
    # 결과 출력
    # print("시설명 벡터화 결과:", facility_embeddings)
    # print("위치설명 벡터화 결과:", location_embeddings)
    # print("서비스설명 벡터화 결과:", service_embeddings)
    # print("층정보 벡터화 결과:", floor_embeddings)

# ### 엑셀 파일의 데이터를 벡터화 -> chromadb에 저장 ###
# # 1. 엑셀 파일 로드
# # 2. 벡터화
# # 3. chromadb 저장

# import pandas as pd
# from transformers import AutoTokenizer, AutoModel
# from langchain.embeddings import HuggingFaceEmbeddings
# from chromadb_connection import store_vectors_in_chroma, mqtt_publish_message  # mqtt_publish_message 추가
# from sklearn.metrics.pairwise import cosine_similarity

# # 엑셀 파일 로드 함수
# def load_excel(file_path):
#     # 엑셀 파일을 DataFrame으로 읽기
#     df = pd.read_excel(file_path)
    
#     print("엑셀 데이터 정보:")
#     print(df.info())
    
#     return df

# # 텍스트 벡터화 함수
# def vectorize_data(df):
#     # 한국어 모델을 사용하여 임베딩
#     tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
#     model = AutoModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
    
#     # 모델을 HuggingFaceEmbeddings에 통합
#     embeddings = HuggingFaceEmbeddings(model_name="monologg/koelectra-base-v3-discriminator")
    
#     # '내용', '역할 및 기능', '기타' 컬럼 결합
#     texts = df['내용'].fillna('') + " " + df['역할 및 기능'].fillna('') + " " + df['기타'].fillna('')
    
#     # 벡터화 진행
#     print("벡터화를 시작합니다...")
#     vectors = embeddings.embed_documents(texts)
    
#     return vectors, texts

# # 벡터화 결과를 비교하는 함수
# def test_vectorization(vectors, texts):
#     # 첫 번째 벡터와 두 번째 벡터의 유사도 계산
#     similarity = cosine_similarity([vectors[0]], [vectors[1]])
#     print(f"첫 번째와 두 번째 벡터 간의 유사도: {similarity[0][0]}")

#     # 예시: 주어진 텍스트 중 첫 번째 텍스트와 유사한 텍스트를 찾기
#     print("\n유사도 높은 텍스트 찾기 (첫 번째 텍스트와 유사한 것):")
#     query_vector = vectors[0]  # 첫 번째 텍스트 벡터
#     similarities = cosine_similarity([query_vector], vectors)  # 첫 번째 텍스트와 나머지 텍스트들의 유사도 계산
#     most_similar_index = similarities.argmax()  # 유사도가 가장 높은 텍스트 인덱스
#     print(f"가장 유사한 텍스트: {texts[most_similar_index]}")
#     print(f"유사도 점수: {similarities[0][most_similar_index]}")
    
# # 메인 실행 함수
# def main():
#     file_path = "C:\\Users\\SSAFY\\Desktop\\LLM\\병원정보.xlsx"  # 엑셀 파일 경로
    
#     # 엑셀 데이터 로드
#     df = load_excel(file_path)
    
#     # 데이터 벡터화
#     vectors, texts = vectorize_data(df)
    
#     # 벡터화 확인
#     test_vectorization(vectors, texts)
    
#     # chromadb 연결 및 벡터 저장
#     collection = store_vectors_in_chroma(vectors, texts)
    
#     # 벡터화 및 저장 완료 메시지를 MQTT로 전송
#     mqtt_publish_message("chroma/status", "벡터화 및 Chroma DB에 저장 완료!")
    
#     print("벡터화 및 Chroma DB에 저장이 완료되었습니다.")

# # 실행
# if __name__ == "__main__":
#     main()

import pandas as pd
from transformers import AutoTokenizer, AutoModel
from langchain_community.embeddings import HuggingFaceEmbeddings  # 변경된 임포트

# 엑셀 파일 로드 함수 (JSON 데이터를 받는 경우도 처리 가능)
def load_excel(file_path):
    # 엑셀 파일을 DataFrame으로 읽기
    df = pd.read_excel(file_path)
    print("엑셀 데이터 정보:")
    print(df.info())
    return df

# 텍스트 벡터화 함수
def vectorize_data(df):
    # 한국어 모델을 사용하여 임베딩
    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    model = AutoModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
    
    # 모델을 HuggingFaceEmbeddings에 통합
    embeddings = HuggingFaceEmbeddings(model_name="monologg/koelectra-base-v3-discriminator")
    
    # '내용', '역할 및 기능', '기타' 컬럼 결합
    texts = df['내용'].fillna('') + " " + df['역할 및 기능'].fillna('') + " " + df['기타'].fillna('')
    
    # 벡터화 진행
    print("벡터화를 시작합니다...")
    vectors = embeddings.embed_documents(texts)
    
    return vectors, texts

# 벡터화 결과를 비교하는 함수 (디버그 용도)
def test_vectorization(vectors, texts):
    # 첫 번째 벡터와 두 번째 벡터의 유사도 계산
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity([vectors[0]], [vectors[1]])
    print(f"첫 번째와 두 번째 벡터 간의 유사도: {similarity[0][0]}")

    # 예시: 주어진 텍스트 중 첫 번째 텍스트와 유사한 텍스트를 찾기
    print("\n유사도 높은 텍스트 찾기 (첫 번째 텍스트와 유사한 것):")
    query_vector = vectors[0]  # 첫 번째 텍스트 벡터
    similarities = cosine_similarity([query_vector], vectors)  # 첫 번째 텍스트와 나머지 텍스트들의 유사도 계산
    most_similar_index = similarities.argmax()  # 유사도가 가장 높은 텍스트 인덱스
    print(f"가장 유사한 텍스트: {texts[most_similar_index]}")
    print(f"유사도 점수: {similarities[0][most_similar_index]}")

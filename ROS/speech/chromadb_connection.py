# ### chromadb 통신 ###

# import chromadb
# from langchain_community.vectorstores import Chroma
# import paho.mqtt.client as mqtt

# # Chroma DB에 벡터 저장 함수
# def store_vectors_in_chroma(vectors, texts):
#     # EC2 ChromaDB 서버에 연결
#     client = chromadb.Client()
    
#     # 컬렉션 생성
#     collection = client.create_collection(name="vectorization_test")
    
#     # 벡터와 텍스트를 컬렉션에 추가
#     collection.add(vectors=vectors, metadatas=texts)
    
#     print(f"Chroma DB에 {len(vectors)}개의 벡터가 저장되었습니다.")
    
#     return collection

# # MQTT로 메시지 전송 함수
# def mqtt_publish_message(topic, message):
#     # MQTT 클라이언트 생성
#     client = mqtt.Client()
    
#     # 서버 연결 (EC2의 Mosquitto 브로커 주소)
#     client.connect("localhost", 1883, 60)  # EC2에서 Mosquitto 브로커 연결
    
#     # 메시지 전송
#     client.publish(topic, message)
#     print(f"MQTT 메시지 전송: {message}")

#     # 연결 종료
#     client.disconnect()

# # # LLM 테스트: Chroma DB에서 벡터 검색 함수
# # def test_llm_with_chroma(collection, query, n_results=3):
# #     # Chroma에서 벡터 기반 검색을 통해 유사한 텍스트 찾기
# #     print(f"질문: {query}")
# #     results = collection.query(query, n_results=n_results)  # 유사도 높은 상위 n개의 텍스트를 반환
    
# #     # 결과 출력
# #     print("LLM의 응답에 사용된 유사한 텍스트:")
# #     for result in results['metadatas']:
# #         print(result)
    
# #     # LLM 호출 (예시: llama 모델, 실제로는 LLM을 호출하는 코드로 대체)
# #     llama_response = "이 부분은 LLM이 생성한 응답입니다."  # 예시 응답
    
# #     return llama_response

import chromadb
from langchain_community.vectorstores import Chroma  # 변경된 임포트

# Chroma DB에 벡터 저장 함수
def store_vectors_in_chroma(vectors, texts):
    # EC2 ChromaDB 서버에 연결
    client = chromadb.Client()

    # 컬렉션 생성
    collection = client.create_collection(name="vectorization_test")

    # 벡터와 텍스트를 컬렉션에 추가
    collection.add(vectors=vectors, metadatas=texts)

    print(f"Chroma DB에 {len(vectors)}개의 벡터가 저장되었습니다.")
    return collection
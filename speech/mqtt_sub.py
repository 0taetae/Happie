### 데이터 수신 ###

import paho.mqtt.client as mqtt
import json  
import chromadb
import numpy as np  # 수학적 계산 및 배열 처리를 위한 라이브러리
import uuid   # 고유한 ID를 생성하기 위한 라이브러리

# Chromadb 클라이언트 설정
client = chromadb.HttpClient(host='', port=8000)

# MQTT 로커 설정
BROKER = ""
PORT = 1883
TOPIC = "chatbot/chromadb"

USERNAME = ""
PASSWORD = ""

# 메시지 수신 후 chromadb에 저장하는 함수
def store_in_chromadb(message):
    client.delete_collection("hospital_info")
    collection = client.create_collection("hospital_info")
    
    # 메시지에서 필요한 데이터를 가져옵니다.
    facility_names = message["시설명"]
    location_descriptions = message["위치설명"]
    service_descriptions = message["서비스설명"]
    floor_info = message["층정보"]
    
    facility_embeddings = message["facility_embeddings"]
    location_embeddings = message["location_embeddings"]
    service_embeddings = message["service_embeddings"]
    floor_embeddings = message["floor_embeddings"]
    
    # 데이터 형식이 리스트인 경우 numpy 배열로 변환
    if isinstance(facility_embeddings, list):
        facility_embeddings = np.array(facility_embeddings)
    if isinstance(location_embeddings, list):
        location_embeddings = np.array(location_embeddings)  # 리스트를 numpy 배열로 변환
    if isinstance(service_embeddings, list):
        service_embeddings = np.array(service_embeddings)
    if isinstance(floor_embeddings, list):
        floor_embeddings = np.expand_dims(np.array(floor_embeddings), axis=1)  #2D 배열로 변환
    
    # 여러 임베딩을 병합하여 하나의 배열로 만듦    
    embeddings = np.hstack([facility_embeddings, location_embeddings, service_embeddings, floor_embeddings])
    
    # 저장할 데이터 형식 설정
    # ChromaDB에 저장할 문서 데이터
    documents = facility_names    # 문서 내용은 시설 이름으로 설정
    
    # 각 문서에 대한 메타데이터 생성
    metadatas = []
    for i in range(len(facility_names)):
        metadatas.append({
            "위치설명": location_descriptions[i],
            "서비스설명": service_descriptions[i],
            "층정보": floor_info[i]
        })
    
    # 각 문서에 대한 고유 ID 생성
    ids = [str(uuid.uuid4()) for _ in facility_names]
    
    # 데이터를 컬렉션에 추가
    collection.add(
        ids=ids,
        documents=documents, 
        metadatas=metadatas, 
        embeddings=embeddings.tolist()  # numpy 배열을 리스트로 변환하여 저장
    )
    
    print("데이터가 chromadb에 저장되었습니다.")

# MQTT 메시지 수신 콜백 함수
def on_message(client, userdata, msg):
    # 수신한 메시지 처리
    print(f"수신한 메시지: {msg.payload.decode()}")
    
    # JSON 형식의 메시지 파싱
    message = json.loads(msg.payload.decode())
    
    #수신한 데이터를 chromadb에 저장
    store_in_chromadb(message)

# MQTT 연결 시 콜백 함수
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe(TOPIC)

# MQTT 구독 시작 함수
def mqtt_subscribe():
    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, 60)
    client.loop_forever()

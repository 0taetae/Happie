### 벡터 데이터 발행 ###

import paho.mqtt.client as mqtt
import json
from vectorization import vectorize_data  # 이미 구현된 벡터화 코드 import
from excel_load import load_excel

# MQTT 브로커 설정
BROKER = "j12e103.p.ssafy.io"  # MQTT 브로커 주소
PORT = 1883  # MQTT 기본 포트
TOPIC = "chatbot/chromadb"  # 메시지를 발행할 주제

# MQTT 사용자 인증 정보 (아이디와 비밀번호)
USERNAME = ""  # 아이디
PASSWORD = ""  # 비밀번호

# 발행할 메시지 생성 함수
def prepare_message(df):
    # 벡터화된 데이터 준비
    facility_embeddings, location_embeddings, service_embeddings, floor_embeddings = vectorize_data(df)
    
    # 데이터를 JSON 형식으로 변환하여 전송
    message = {
        "시설명": df['시설명'].tolist(),
        "위치설명": df['위치설명'].tolist(),
        "서비스설명": df['서비스설명'].tolist(),
        "층정보": df['층정보'].tolist(),
        "facility_embeddings": facility_embeddings.tolist(),  # 벡터화된 데이터
        "location_embeddings": location_embeddings.tolist(),
        "service_embeddings": service_embeddings.tolist(),
        "floor_embeddings": floor_embeddings.tolist()
    }
    
    return json.dumps(message)  # JSON 형식으로 반환

# MQTT 클라이언트 설정 및 연결
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    
    # 엑셀 파일 로드
    df = load_excel()
    
    # 메시지 준비
    message = prepare_message(df)
    
    # 메시지 발행
    print("발행 중...")
    client.publish(TOPIC, message)
    print("메시지가 발행되었습니다.")
    
    # 연결 종료
    client.disconnect()

def mqtt_publish():
    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.on_connect = on_connect
    client.connect(BROKER, PORT, 60)
    # client.loop_forever()
    client.loop_start() # 비동기 실행 (메시지 발행 후 대기)

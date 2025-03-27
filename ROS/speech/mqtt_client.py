import paho.mqtt.client as mqtt
import json
import pandas as pd  # DataFrame을 사용하기 위해 pandas를 임포트
from vectorization import vectorize_data  # 벡터화 함수
from chromadb_connection import store_vectors_in_chroma  # Chroma DB에 벡터 저장

# Mosquitto 브로커와의 연결 설정
BROKER_HOST = "j12e103.p.ssafy.io"  # EC2 서버의 IP 주소
BROKER_PORT = 1883  # 기본 Mosquitto 포트
TOPIC = "hospital_data_topic"  # 구독할 MQTT 토픽

# MQTT 인증 정보 (아이디와 비밀번호)
MQTT_USERNAME = "happie_mqtt_user"
MQTT_PASSWORD = "gkstkfckdl0411!"

# 연결 성공 시 실행되는 콜백 함수
def on_connect(client, userdata, flags, rc):
    print(f"연결 상태: {rc}")
    if rc == 0:  # 연결이 성공했을 때
        print("MQTT 브로커에 성공적으로 연결되었습니다.")
        client.subscribe(TOPIC)  # 토픽 구독 시작
    else:
        print("MQTT 브로커 연결 실패")

# 메시지를 수신했을 때 실행되는 콜백 함수
def on_message(client, userdata, msg):
    print(f"메시지 수신: {msg.payload.decode()}")
    
    try:
        # 수신된 메시지 처리 (예: JSON 형태로 받았을 경우)
        data = json.loads(msg.payload.decode())  # 메시지를 JSON으로 변환
        print(f"수신된 데이터: {data}")

        # JSON 데이터를 DataFrame으로 변환
        df = pd.DataFrame([data])  # data는 하나의 레코드이므로 리스트로 감싸서 DataFrame으로 변환
        print("DataFrame으로 변환된 데이터:")
        print(df)

        # 데이터 벡터화
        vectors, texts = vectorize_data(df)  # 벡터화
        print(f"벡터화된 벡터: {vectors[:2]}")  # 첫 번째 두 벡터만 출력하여 확인
        
        # Chroma DB에 벡터 저장
        store_vectors_in_chroma(vectors, texts)
        print("벡터화된 데이터를 Chroma DB에 저장했습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

# MQTT 클라이언트 설정
client = mqtt.Client()

# MQTT 인증 설정 (아이디와 비밀번호)
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

# 연결 콜백 함수 설정
client.on_connect = on_connect

# 메시지 수신 콜백 함수 설정
client.on_message = on_message

# Mosquitto 브로커에 연결
client.connect(BROKER_HOST, BROKER_PORT, 60)

# MQTT 클라이언트 루프 시작 (수신 대기)
client.loop_forever()

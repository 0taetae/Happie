### 대화형 챗봇 시스템 MQTT 통신 ###
# 1. MQTT Broker의 음성 파일을 구독
# 2. STT
# 3. 텍스트 파일(LLM 결과물)을 MQTT Broker로 발행

import paho.mqtt.client as mqtt
import io
import base64
from stt import transcribe_stt # 음성을 텍스트로 변환

# MQTT 설정
BROKER = ""
PORT = 1883
USERNAME = ""
PASSWORD = ""

TOPIC_SUBSCRIBE = "user/chatbot/request"  # 음성 데이터 구독 토픽
TOPIC_PUBLISH = "chatbot/response"       # 변환된 텍스트 발행 토픽

# MQTT 메시지 수신 콜백 함수
def on_message(client, userdata, msg):
    print(f"수신한 메시지 (topic: {msg.topic})")
    try:
        # Base64 인코딩 된 음성 데이터 디코딩
        audio_data = base64.b64decode(msg.payload)

        # 음성 데이터를 메모리에서 처리 (BytesIO로 변환)
        audio_buffer = io.BytesIO(audio_data)
        print("음성 파일 처리 중...")

        # STT를 통해 텍스트로 변환
        transcribed_text = transcribe_stt(audio_buffer)
        if transcribed_text:
            print("변환된 텍스트:", transcribed_text)

            # 변환된 텍스트를 MQTT로 발행
            client.publish(TOPIC_PUBLISH, transcribed_text)
            print("텍스트가 MQTT 브로커에 발행되었습니다.")
    except Exception as e:
        print(f"Error during message handling: {e}")


# MQTT 연결 콜백 함수
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    if rc == 0:
        print("MQTT 브로커 연결 성공")
        client.subscribe(TOPIC_SUBSCRIBE)  # 음성 데이터 구독 시작
    else:
        print(f"MQTT 브로커 연결 실패 (코드: {rc})")

# MQTT 구독 및 발행 시작 함수
def start_mqtt():
    client = mqtt.Client(protocol=mqtt.MQTTv311)
    client.username_pw_set(USERNAME, PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(BROKER, PORT, 60)
        print("MQTT 브로커에 연결 중...")
        client.loop_forever() # 이벤트 기반 시스템 동작 : MQTT 메시지가 들어오면 on_message 호출
    except Exception as e:
        print(f"MQTT 브로커 연결 오류: {e}")

if __name__ == "__main__":
    start_mqtt()

import paho.mqtt.client as mqtt
import io

# MQTT 설정
BROKER = ""
PORT = 1883
USERNAME = ""
PASSWORD = ""

TOPIC_PUBLISH = "chatbot/response"  # 변환된 텍스트 발행 토픽

# MQTT 연결 콜백 함수
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    if rc == 0:
        print("MQTT 브로커 연결 성공")
    else:
        print(f"MQTT 브로커 연결 실패 (코드: {rc})")

# MQTT 텍스트 발행 함수
def publish_text(text):
    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.on_connect = on_connect

    try:
        client.connect(BROKER, PORT, 60)
        client.publish(TOPIC_PUBLISH, text)
        print("텍스트가 MQTT 브로커에 발행되었습니다.")
    except Exception as e:
        print(f"MQTT 브로커 연결 오류: {e}")

# 텍스트 발행 예시
if __name__ == "__main__":
    example_text = "이것은 변환된 텍스트 예시입니다."
    publish_text(example_text)

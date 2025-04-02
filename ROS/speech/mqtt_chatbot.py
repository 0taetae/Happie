### 대화형 챗봇 시스템 MQTT 통신 ###
# 1. MQTT Broker의 음성 파일을 구독
# 2. STT
# 3. 텍스트 파일(LLM 결과물)을 MQTT Broker로 발행

import paho.mqtt.client as mqtt
import io
import base64
import json
from stt import transcribe_stt # 음성을 텍스트로 변환
from search_vector import search_hospital_info  # 벡터 검색
from prompting import generate_response  # 프롬프트 생성

# MQTT 설정
BROKER = "j12e103.p.ssafy.io"
PORT = 1883
USERNAME = ""
PASSWORD = ""

TOPIC_SUBSCRIBE = "user/chatbot/request"  # 음성 데이터 구독 토픽
TOPIC_PUBLISH = "chatbot/response"       # 변환된 텍스트 발행 토픽

# ✅ 올바른 Base64 데이터 체크 함수
def is_base64(sb):
    try:
        base64.b64decode(sb, validate=True)  # Base64로 변환 가능 여부 확인
        return True
    except Exception:
        return False

# MQTT 메시지 수신 콜백 함수
def on_message(client, userdata, msg):
    print(f"📩 수신한 메시지 (topic: {msg.topic}), 크기: {len(msg.payload)} bytes")

    try:
        # 메시지 크기가 작으면 텍스트로 간주 (예: 1KB 미만)
        if len(msg.payload) < 1024:
            try:
                transcribed_text = msg.payload.decode('utf-8')  # 텍스트 데이터인지 확인
                print("📄 텍스트 데이터 수신:", transcribed_text)
            except UnicodeDecodeError:
                print("⚠️ UTF-8 해석 실패, 데이터 형식 확인 필요.")
                return
        else:
            # 메시지 크기가 크면 바이너리 음성 데이터로 간주
            print("🎵 바이너리 음성 데이터 감지... 처리 중")
            try:
                audio_data = msg.payload.decode('utf-8')  # Base64로 인코딩된 음성 데이터가 문자열로 들어옵니다.
                
                # Base64 패딩 추가 (4의 배수로 맞추기 위해)
                padding = '=' * (4 - len(audio_data) % 4)  # 패딩 추가
                audio_data += padding

                # Base64 디코딩
                audio_data = base64.b64decode(audio_data)  # Base64 디코딩
                audio_buffer = io.BytesIO(audio_data)  # BytesIO로 변환
                print("음성 파일 처리 중...")

                # STT로 변환
                transcribed_text = transcribe_stt(audio_buffer)
                if not transcribed_text:
                    print("음성에서 텍스트 변환 실패")
                    return
                print("변환된 텍스트:", transcribed_text)

            except Exception as e:
                print(f"❌ 음성 처리 중 오류 발생: {e}")
                return

        # 이후에는 음성 또는 텍스트 데이터에 대한 동일한 후처리 로직
        search_results = search_hospital_info(transcribed_text)
        print(f"🔍 검색된 병원 정보 : {search_results}")

        # LLM 응답 생성
        response_text = generate_response(transcribed_text, search_results)
        print(f"🤖 LLM 응답 : {response_text}")

        # # "5층"이 포함되어 있는지 확인
        facility_name = ""
        if "5층" in response_text:
            for result in search_results:
                if result.get("floor_info") == "5층":
                    facility_name = result.get("facility_name", "")
                    break
            response_text += " 안내를 시작할까요?"

        # # 사용자 질문과 LLM 응답을 JSON 객체로 묶어서 전송
        message_data = {
            "request": transcribed_text,
            "response": response_text,
            "facility": facility_name
        }

        # # JSON 형식으로 변환
        message_json = json.dumps(message_data, ensure_ascii=False)

        # # 변환된 JSON 메시지를 MQTT로 발행
        client.publish(TOPIC_PUBLISH, message_json)
        print("✅ 응답이 MQTT 브로커에 발행되었습니다.")
        
    except Exception as e:
        print(f"⚠️ Error during message handling: {e}")


# MQTT 연결 콜백 함수
def on_connect(client, userdata, flags, rc):
    print(f"🔗 Connected with result code {rc}")
    if rc == 0:
        print("✅ MQTT 브로커 연결 성공")
        client.subscribe(TOPIC_SUBSCRIBE)  # 음성 데이터 구독 시작
    else:
        print(f"❌ MQTT 브로커 연결 실패 (코드: {rc})")

# MQTT 구독 및 발행 시작 함수
def start_mqtt():
    client = mqtt.Client(protocol=mqtt.MQTTv311)
    client.username_pw_set(USERNAME, PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(BROKER, PORT, 60)
        print("🔄 MQTT 브로커에 연결 중...")
        client.loop_forever()  # 이벤트 기반 시스템 동작 : MQTT 메시지가 들어오면 on_message 호출
    except Exception as e:
        print(f"❌ MQTT 브로커 연결 오류: {e}")

if __name__ == "__main__":
    start_mqtt()
import paho.mqtt.client as mqtt
import io
import base64
import json
from stt import transcribe_stt
from search_vector import search_hospital_info
from prompting import generate_response, clear_history
import threading

class MQTTChatbot:
    def __init__(self):
        self.BROKER = "j12e103.p.ssafy.io"
        self.PORT = 1883

        self.TOPIC_SUBSCRIBE = "user/chatbot/request"
        self.TOPIC_PUBLISH = "chatbot/response"

        self.TIMEOUT_SECONDS = 60 * 3  # 3분 후 history 초기화
        self.history_reset_timer = None  # 타이머 변수

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv311)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def start(self):
        """MQTT 연결 시작"""
        try:
            self.client.connect(self.BROKER, self.PORT, 60)
            print("🔄 MQTT 브로커에 연결 중...")
            self.client.loop_forever()
        except Exception as e:
            print(f"❌ MQTT 브로커 연결 오류: {e}")

    def on_connect(self, client, userdata, flags, reason_code, properties):
        """MQTT 연결 이벤트"""
        if reason_code == 0:
            print("✅ MQTT 브로커 연결 성공")
            client.subscribe(self.TOPIC_SUBSCRIBE)
        else:
            print(f"❌ MQTT 브로커 연결 실패 (코드: {reason_code})")

    def on_message(self, client, userdata, msg):
        """MQTT 메시지 수신 이벤트"""
        # 기존 타이머 취소
        if self.history_reset_timer:
            self.history_reset_timer.cancel()

        print(f"📩 수신한 메시지 (topic: {msg.topic}), 크기: {len(msg.payload)} bytes")

        try:
            # 메시지 크기가 작으면 텍스트로 간주
            if len(msg.payload) < 1024:
                try:
                    transcribed_text = msg.payload.decode('utf-8')
                    print("📄 텍스트 데이터 수신:", transcribed_text)
                except UnicodeDecodeError:
                    print("⚠️ UTF-8 해석 실패, 데이터 형식 확인 필요.")
                    return
            else:
                # 바이너리 음성 데이터 처리
                transcribed_text = self.process_audio(msg.payload)

            if not transcribed_text:
                return

            search_results = search_hospital_info(transcribed_text)
            response_text = generate_response(transcribed_text, search_results)

            facility_name = ""
            if "5층" in response_text:
                for result in search_results:
                    if result.get("floor_info") == "5층":
                        facility_name = result.get("facility_name", "")
                        break
                response_text += " 안내를 시작할까요?"

            message_data = {
                "request": transcribed_text,
                "response": response_text,
                "facility": facility_name
            }

            message_json = json.dumps(message_data, ensure_ascii=False)
            client.publish(self.TOPIC_PUBLISH, message_json)
            print("✅ 응답이 MQTT 브로커에 발행되었습니다.")

            # 새로운 타이머 시작
            self.history_reset_timer = threading.Timer(self.TIMEOUT_SECONDS, self.reset_history)
            self.history_reset_timer.start()

        except Exception as e:
            print(f"⚠️ 메시지 처리 중 오류 발생: {e}")

    def process_audio(self, audio_payload):
        """바이너리 음성 데이터를 텍스트로 변환"""
        try:
            audio_data = audio_payload.decode('utf-8')

            # Base64 패딩 보정
            padding = '=' * (4 - len(audio_data) % 4)
            audio_data += padding

            # Base64 디코딩
            audio_data = base64.b64decode(audio_data)
            audio_buffer = io.BytesIO(audio_data)

            print("음성 파일 처리 중...")
            return transcribe_stt(audio_buffer)

        except Exception as e:
            print(f"❌ 음성 처리 중 오류 발생: {e}")
            return None

    def reset_history(self):
        """대화 기록 초기화"""
        print("⏳ 대화 기록 초기화 (사용자 응답 없음)")
        clear_history()
        self.history_reset_timer = None


if __name__ == "__main__":
    chatbot = MQTTChatbot()
    chatbot.start()

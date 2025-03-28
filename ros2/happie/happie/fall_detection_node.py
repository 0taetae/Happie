import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import torch
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import json
import base64
import time
import boto3
from datetime import datetime
import config       // S3, MQTT 설정

S3_BUCKET = config.S3_BUCKET  # 실제 버킷 이름
S3_FOLDER = config.S3_FOLDER

s3_client = boto3.client(
    's3',
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
    region_name='ap-northeast-2'
)

# MQTT 설정
BROKER = config.BROKER
PORT = config.PORT
TOPIC = "fall_detection"
USERNAME = config.USERNAME
PASSWORD = config.PASSWORD

class FallDetectionNode(Node):
    def __init__(self):
        super().__init__('fall_detection_node')

        # 이미지 구독
        self.subscriber = self.create_subscription(
            CompressedImage,        # 메시지 타입
            '/image_jpeg/compressed',   # 구독할 토픽 이름
            self.image_callback,        # 콜백 함수
            10      # 큐 사이즈
        )

        self.last_sent_time = 0     # 3초 주기로 전송하기 위해서 마지막 MQTT 전송 시간을 저장

        # YOLO 모델 로드
        model_path = r'C:\Users\SSAFY\Desktop\project\mobility-smarthome-skeleton\yolov5\runs\train\exp10\weights\best.pt'
        yolov5_dir = r'C:\Users\SSAFY\Desktop\project\mobility-smarthome-skeleton\yolov5'

        self.model = torch.hub.load(
            yolov5_dir,     # YOLOv5 디렉토리
            'custom',       # 사용할 커스텀 모델
            path=model_path,        # 모델 경로
            source='local'      # 로컬에서 로드
        )
        self.model.conf = 0.7  # 신뢰도 임계값 (0.6 이상일 때만 감지된 것으로 처리함)

        # MQTT 클라이언트 설정
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.username_pw_set(USERNAME, PASSWORD)  # 인증 정보 설정
        self.mqtt_client.on_connect = self.on_connect         # 연결 시 콜백 함수 설정
        self.mqtt_client.connect(BROKER, PORT, 60)            # 브로커에 연결
        self.mqtt_client.loop_start()                         # MQTT 클라이언트 루프 시작 (비동기)

    def on_connect(self, client, userdata, flags, rc):
        self.get_logger().info(f"MQTT Connected with result code {rc}")     # 연결 성공 시 로그

    def image_callback(self, msg):
        # 압축된 이미지 데이터를 numpy 배열로 변환
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)      # OpenCV 이미지 디코딩 - 테스트할 때 사용(삭제 예정)

        # 객체 탐지 수행
        results = self.model(frame)

        # 낙상 여부
        fall_detected = False

        # 탐지된 객체들 반복 처리
        for *xyxy, conf, cls in results.xyxy[0]:
            label = self.model.names[int(cls)]      # 클래스 라벨 이름 추출 ('fall', 'whelchair', 'intravenous' 등등 있음)

            # 낙상 감지
            if label == 'fall':
                fall_detected = True

                # 낙상 이미지를 빨간색 박스로 표시
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                              (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)

                # 낙상 텍스트와 신뢰도 표시
                cv2.putText(frame, f'FALL {conf:.2f}',
                            (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # 현재 시간 체크
        current_time = time.time()

        # 이미지가 너무 많이 전송되지 않도록 3초마다 전송
        if fall_detected and (current_time - self.last_sent_time > 3):
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()

            # s3에 저장할 고유한 파일명 생성
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_filename = f"{S3_FOLDER}/fall_{timestamp}.jpg"

            try:
                # s3에 이미지 업로드
                s3_client.put_object(
                    Bucket=S3_BUCKET,
                    Key=image_filename,
                    Body=image_bytes,
                    ContentType='image/jpeg'
                )
                # 이미지 url 생성
                image_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{image_filename}"

                payload = {
                    "event": "fall",
                    "image_url": image_url
                }

                # MQTT 브로커에 메시지 발행
                self.mqtt_client.publish(TOPIC, json.dumps(payload))
                self.get_logger().info("image sent to MQTT broker")

                self.last_sent_time = current_time
            except Exception as e:
                self.get_logger().error(f"Failed to upload image: {e}")

        # fix: 디버깅용으로 이미지 화면에 출력
        cv2.imshow("Fall", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = FallDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

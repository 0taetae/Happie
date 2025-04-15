import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import threading
import torch
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import json
import base64
import time
import boto3
from datetime import datetime
from .config import YOLOV5_DIR, MODEL_PATH
from . import config
import os
from std_msgs.msg import Bool

print(torch.cuda.is_available())     # True여야 함
# print(torch.cuda.get_device_name(0)) # GPU 이름 출력됨

# S3 설정
S3_BUCKET = config.S3_BUCKET
S3_FOLDER = config.S3_FOLDER

s3_client = boto3.client(
    's3',
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
    region_name='ap-northeast-2'
)

# MQTT 설정
BROKER = config.MQTT_CONFIG["BROKER"]
PORT = config.MQTT_CONFIG["PORT"]
USERNAME = config.MQTT_CONFIG["USERNAME"]
PASSWORD = config.MQTT_CONFIG["PASSWORD"]
TOPIC = "fall_detection"

class FallDetectionNode(Node):
    def __init__(self):
        super().__init__('fall_detection_node')

        self.subscriber = self.create_subscription(
            CompressedImage,
            '/image_jpeg/compressed',
            self.image_callback,
            30
        )

        self.last_sent_time = 0
        self.prev_fall_state = False

        CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
        yolov5_dir = YOLOV5_DIR
        model_path = MODEL_PATH

        self.model = torch.hub.load(
            yolov5_dir,
            'custom',
            path=model_path,
            source='local'
        )

        # GPU가 사용 가능하면 모델을 CUDA로 옮김
        if torch.cuda.is_available():
            self.model.to('cuda')
            self.get_logger().info("Using GPU for inference")
        else:
            self.get_logger().info("Using CPU for inference")

        self.model.conf = 0.7
        self.fall_pub = self.create_publisher(Bool,'/fall_detected',1) # 낙상감지 

        # MQTT 설정
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.username_pw_set(USERNAME, PASSWORD)
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.connect(BROKER, PORT, 60)
        self.mqtt_client.loop_start()

        # 동시 처리를 위한 락
        self.lock = threading.Lock()
        self.is_processing = False 

    def on_connect(self, client, userdata, flags, rc):
        self.get_logger().info(f"MQTT Connected with result code {rc}")

    def image_callback(self, msg):
        # 비동기 처리: 이미 처리 중이면 무시
        if self.is_processing:
            return
        
        with self.lock:
            self.is_processing = True

            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            results = self.model(frame)
            fall_detected = False

            for *xyxy, conf, cls in results.xyxy[0]:
                label = self.model.names[int(cls)]

                if label == 'fall':
                    fall_detected = True

                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                                  (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
                    cv2.putText(frame, f'FALL {conf:.2f}',
                                (int(xyxy[0]), int(xyxy[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            if fall_detected != self.prev_fall_state:
                self.fall_pub.publish(Bool(data=True))
                self.prev_fall_state = fall_detected
                self.get_logger().info(f"fall_detected status publish: {fall_detected}")
            
            # 이미지 저장 및 MQTT 전송
            current_time = time.time()

            if fall_detected and (current_time - self.last_sent_time > 30):
                _, buffer = cv2.imencode('.jpg', frame)
                image_bytes = buffer.tobytes()

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                image_filename = f"{S3_FOLDER}/fall_{timestamp}.jpg"

                try:
                    s3_client.put_object(
                        Bucket=S3_BUCKET,
                        Key=image_filename,
                        Body=image_bytes,
                        ContentType='image/jpeg'
                    )
                    image_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{image_filename}"

                    payload = {
                        "event": "fall",
                        "image_url": image_url
                    }

                    self.get_logger().info(f"Publishing payload: {payload}")
                    self.mqtt_client.publish(TOPIC, json.dumps(payload))
                    self.get_logger().info("image sent to MQTT broker")

                    self.last_sent_time = current_time
                    
                except Exception as e:
                    self.get_logger().error(f"Failed to upload image: {e}")

            cv2.imshow("Fall", frame)
            cv2.waitKey(5)

            self.is_processing = False


def main(args=None):
    rclpy.init(args=args)
    node = FallDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
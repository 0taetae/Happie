import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32
from ssafy_msgs.msg import HandControl
import torch
import cv2
import numpy as np
import time
import pymysql
import re
import os
import json
import threading
import paho.mqtt.client as mqtt
from . import config  # DB 설정 불러오기

# MQTT 설정
BROKER = config.MQTT_CONFIG["BROKER"]
PORT = config.MQTT_CONFIG["PORT"]
USERNAME = config.MQTT_CONFIG["USERNAME"]
PASSWORD = config.MQTT_CONFIG["PASSWORD"]
TOPIC = "robot/equipment"

class EquipmentDetectionNode(Node):
    def __init__(self):
        super().__init__('equipment_detection_node')

        # ROS Subscriber & Publisher
        self.subscriber = self.create_subscription(
            CompressedImage,
            '/image_jpeg/compressed',
            self.image_callback,
            50
        )
        self.hand_control_pub = self.create_publisher(
            HandControl,
            '/hand_control',
            10
        )
        self.equipment_detected_pub = self.create_publisher(
            Int32,
            '/equipment_detected',
            10
        )

        # /is_equipment_detection_on 토픽 구독
        self.subscription = self.create_subscription(
            Int32,
            '/is_equipment_detection_on',
            self.is_equipment_detection_on_callback,
            10  # QoS Depth
        )


        # 시간
        self.last_infer_time = time.time()

        # YOLOv5 모델 로드
        yolov5_dir = config.YOLOV5_DIR
        model_path = config.MODEL_PATH
        print(config.YOLOV5_DIR)


        self.model = torch.hub.load(
            yolov5_dir,
            'custom',
            path=model_path,
            source='local'
        )
        self.model.conf = 0.5

        # 추론모드
        self.model.eval()

        # 상태 변수
        self.last_command_time = 0
        self.command_interval = 5.0
        self.current_target = None
        self.current_order_id = None       
        self.current_destination = None
        self.destination_coords = None
        self.target_detected = False  # 기자재 감지 여부
        self.is_processing = False

        # MQTT 설정
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.username_pw_set(USERNAME, PASSWORD)
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        self.mqtt_client.connect(BROKER, PORT, 60)
        self.mqtt_client.loop_start()


    def image_callback(self, msg):
        # print('self.current_order_id:', self.current_order_id)
        # print('self.current_target:', self.current_target)

        if time.time() - self.last_infer_time < 1:  # 0.5초마다만 실행
            return
        self.last_infer_time = time.time()

        if not self.current_target:
            return

        if self.is_processing: 
            return

        self.is_processing = True

        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        results = self.model(frame)

        msg = Int32()
        msg.data = 0
       # 하나만 처리
        for *xyxy, conf, cls in results.xyxy[0]:
            label = self.model.names[int(cls)]

            if label == self.current_target:
                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label.upper()} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                self.target_detected = True

                msg.data = 1  # 예: 감지되었음을 알리는 값
                break  # ✅ 첫 번째만 처리 후 break

        # 결과값 publish        
        # self.get_logger().info(f"📡 Equipment detected, published value: {msg.data}")
        self.equipment_detected_pub.publish(msg)

        # 필요하면 한 번만 보내고 멈추도록
        self.target_detected = False


        self.is_processing = False
        cv2.imshow("Equipment Detection", frame)
        cv2.waitKey(1)

    def is_equipment_detection_on_callback(self, msg):
        print(f'is_equipment_detection_on: {msg.data}')
        # detection 카메라 종료를 위함
        if msg.data == 0:
            self.current_order_id = None
            self.current_target = None
            cv2.destroyAllWindows()
            cv2.waitKey(1)


    # MQTT 관련
    def on_mqtt_connect(self, client, userdata, flags, rc):
        self.get_logger().info("[MQTT] 연결 성공")
        client.subscribe(TOPIC)

    def on_mqtt_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8")
            data = json.loads(payload)
            if msg.topic == TOPIC:
                self.current_order_id = data['no']
                self.current_target = [None, "wheelchair", "intravenous"][self.current_order_id]

        except Exception as e:
            self.get_logger().error(f"[MQTT 처리 에러] {e}")

def main(args=None):
    rclpy.init(args=args)
    node = EquipmentDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

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
TOPIC = "robot/log"

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

        # 상태 변수
        self.last_command_time = 0
        self.command_interval = 5.0
        self.current_target = None
        self.current_order_id = None
        self.current_destination = None
        self.destination_coords = None
        self.target_detected = False  # 기자재 감지 여부
        self.is_processing = False

        # DB 주문 확인 타이머
        self.create_timer(5.0, self.check_orders_from_db)

        # MQTT 설정
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.username_pw_set(USERNAME, PASSWORD)
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        self.mqtt_client.connect(BROKER, PORT, 60)
        self.mqtt_client.loop_start()

    def check_orders_from_db(self):
        try:
            conn = pymysql.connect(
                **config.MYSQL,
                cursorclass=pymysql.cursors.DictCursor
            )
            with conn.cursor() as cursor:
                sql = "SELECT id, place, todo, x, y FROM orders WHERE state = '대기' AND robot = 'robot1' ORDER BY id ASC LIMIT 1"
                cursor.execute(sql)
                row = cursor.fetchone()

                if row:
                    order_id = row['id']
                    place = row['place']
                    todo = row['todo']
                    x = float(row['x'])
                    y = float(row['y'])

                    match = re.match(r'(\w+)\s+전달', todo)
                    if match:
                        equipment_kor = match.group(1)
                        mapping = {
                            '휠체어': 'wheelchair',
                            '링거': 'intravenous'
                        }
                        equipment_eng = mapping.get(equipment_kor)

                        if equipment_eng and order_id != self.current_order_id:
                            self.current_target = equipment_eng
                            self.current_order_id = order_id
                            self.current_destination = place
                            self.destination_coords = (x, y)
                            self.target_detected = False
                            self.get_logger().info(f"[DB 명령 인식] id={order_id}, 대상: {self.current_target}, 목적지: {place}, 좌표: ({x}, {y})")
            conn.close()
        except Exception as e:
            self.get_logger().error(f"[DB 에러] {e}")

    def image_callback(self, msg):
        if not self.current_target:
            return

        if self.is_processing: 
            return

        self.is_processing = True

        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        results = self.model(frame)

        for *xyxy, conf, cls in results.xyxy[0]:
            label = self.model.names[int(cls)]

            if label == self.current_target:
                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label.upper()} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                self.target_detected = True  # ✅ 감지만 하고 집진 않음
            self.is_processing = False

        self.is_processing = False
        # cv2.imshow("Equipment Detection", frame)
        # cv2.waitKey(1)


    def send_hand_control_pickup(self):
        msg = HandControl()
        msg.control_mode = 2  # 집기
        self.hand_control_pub.publish(msg)
        self.get_logger().info("📦 기자재 집기 명령 전송")

    def send_hand_control_release(self):
        msg = HandControl()
        msg.control_mode = 3  # 내려놓기
        self.hand_control_pub.publish(msg)
        self.get_logger().info("📦 기자재 내려놓기 명령 전송")

    def update_order_state_to_in_progress(self, order_id):
        try:
            conn = pymysql.connect(
                **config.MYSQL,
                cursorclass=pymysql.cursors.DictCursor
            )
            with conn.cursor() as cursor:
                sql = "UPDATE orders SET state = '진행 중' WHERE id = %s"
                cursor.execute(sql, (order_id,))
                conn.commit()
                self.get_logger().info(f"[DB 업데이트] id={order_id} → '진행 중'")
            conn.close()
        except Exception as e:
            self.get_logger().error(f"[DB 업데이트 에러] {e}")

    def update_order_state_to_done(self, order_id):
        try:
            conn = pymysql.connect(
                **config.MYSQL,
                cursorclass=pymysql.cursors.DictCursor
            )
            with conn.cursor() as cursor:
                sql = "UPDATE orders SET state = '완료' WHERE id = %s"
                cursor.execute(sql, (order_id,))
                conn.commit()
                self.get_logger().info(f"[DB 업데이트] id={order_id} → '완료'")
            conn.close()

            # 상태 초기화
            self.current_target = None
            self.current_order_id = None
            self.current_destination = None
            self.destination_coords = None
            self.target_detected = False

        except Exception as e:
            self.get_logger().error(f"[DB 업데이트 에러] {e}")

    # MQTT 관련
    def on_mqtt_connect(self, client, userdata, flags, rc):
        self.get_logger().info("[MQTT] 연결 성공")
        client.subscribe(TOPIC)

    def on_mqtt_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            order_id = payload.get("id")
            status = payload.get("status")

            if self.current_order_id is None or int(order_id) != int(self.current_order_id):
                return

            if status == "arrive":
                self.get_logger().info(f"[MQTT] 로봇 도착 (id={order_id}) → 기자재 집기 강행")
                self.send_hand_control_pickup()
                self.update_order_state_to_in_progress(order_id)

            elif status == "finish":
                self.get_logger().info(f"[MQTT] 작업 완료 (id={order_id}) → 내려놓기 수행")
                self.send_hand_control_release()
                self.update_order_state_to_done(order_id)

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

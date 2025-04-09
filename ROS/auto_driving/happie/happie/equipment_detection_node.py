import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from ssafy_msgs.msg import HandControl
import torch
import cv2
import numpy as np
import time
import pymysql
import re
import os
from . import config  # DB 설정 불러오기

class EquipmentDetectionNode(Node):
    def __init__(self):
        super().__init__('equipment_detection_node')

        self.subscriber = self.create_subscription(
            CompressedImage,
            '/image_jpeg/compressed',
            self.image_callback,
            10
        )

        self.hand_control_pub = self.create_publisher(
            HandControl,
            '/hand_control',
            10
        )

        yolov5_dir = YOLOV5_DIR
        model_path = MODEL_PATH

        self.model = torch.hub.load(
            yolov5_dir,
            'custom',
            path=model_path,
            source='local'
        )
        self.model.conf = 0.5

        self.last_command_time = 0
        self.command_interval = 5.0
        self.current_target = None
        self.current_order_id = None
        self.current_destination = None
        self.destination_coords = None

        self.create_timer(5.0, self.check_orders_from_db)

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
                            self.get_logger().info(f"[DB 명령 인식] id={order_id}, 대상: {self.current_target}, 목적지: {place}, 좌표: ({x}, {y})")
            conn.close()
        except Exception as e:
            self.get_logger().error(f"[DB 에러] {e}")

    def image_callback(self, msg):
        if not self.current_target:
            return

        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        results = self.model(frame)

        for *xyxy, conf, cls in results.xyxy[0]:
            label = self.model.names[int(cls)]

            if label == self.current_target:
                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                # HSV 색상 추출 제거됨 👇

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label.upper()} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if time.time() - self.last_command_time > self.command_interval:
                    self.send_hand_control_pickup()
                    self.last_command_time = time.time()
                    self.update_order_state_to_in_progress(self.current_order_id)

        cv2.imshow("Equipment Detection", frame)
        cv2.waitKey(1)

    def send_hand_control_pickup(self):
        msg = HandControl()
        msg.control_mode = 2
        self.hand_control_pub.publish(msg)
        self.get_logger().info(f"{self.current_target} 감지됨 → 픽업 명령 전송")

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
                self.get_logger().info(f"[DB 업데이트] id={order_id} → '진행 중'으로 변경")
            conn.close()

            self.current_target = None
            self.current_order_id = None
            self.current_destination = None
            self.destination_coords = None
        except Exception as e:
            self.get_logger().error(f"[DB 업데이트 에러] {e}")

def main(args=None):
    rclpy.init(args=args)
    node = EquipmentDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

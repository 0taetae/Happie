import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, LaserScan
from ssafy_msgs.msg import HandControl
import torch
import cv2
import numpy as np
import time
import pymysql
import re
import os
from . import config


class EquipmentDetectionNode(Node):
    def __init__(self):
        super().__init__('equipment_detection_node')

        self.subscriber = self.create_subscription(
            CompressedImage,
            '/image_jpeg/compressed',
            self.image_callback,
            80
        )

        self.lidar_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            1
        )

        self.hand_control_pub = self.create_publisher(
            HandControl,
            '/hand_control',
            10
        )

        self.lidar_data = None
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.heading = 0.0

        self.model = torch.hub.load(
            config.YOLOV5_DIR,
            'custom',
            path=config.MODEL_PATH,
            source='local'
        )

        self.model.conf = 0.3
        print("모델 클래스 목록:", self.model.names)

        self.last_command_time = 0
        self.command_interval = 5.0
        self.current_target = 'wheelchair'
        self.current_order_id = None
        self.current_destination = None
        self.destination_coords = None

        self.create_timer(5.0, self.check_orders_from_db)

        self.ranges = []
        self.is_processing = False

    def lidar_callback(self, msg):
        self.lidar_data = msg.ranges
        self.pose_x = msg.range_min
        self.pose_y = msg.scan_time
        self.heading = msg.time_increment
        self.ranges =msg.ranges

    def check_orders_from_db(self):
        return
        try:
            conn = pymysql.connect(**config.MYSQL, cursorclass=pymysql.cursors.DictCursor)
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
                        mapping = {'휠체어': 'wheelchair', '링거': 'intravenous'}
                        equipment_eng = mapping.get(equipment_kor)
                        if equipment_eng and order_id != self.current_order_id:
                            self.current_target = equipment_eng
                            self.current_order_id = order_id
                            self.current_destination = place
                            self.destination_coords = (x, y)
                            print(f"[DB 명령 인식] id={order_id}, 대상: {self.current_target}, 목적지: {place}, 좌표: ({x}, {y})")
            conn.close()
        except Exception as e:
            self.get_logger().error(f"[DB 에러] {e}")

    def image_callback(self, msg):
        if not self.current_target or self.is_processing:
            return

        if self.lidar_data is None:
            print("Lidar 데이터 없음!")
            return

        self.is_processing = True
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        results = self.model(frame)
        print(f"[디버깅] 탐지 결과 개수: {len(results.xyxy[0])}")

        img_h, img_w = frame.shape[:2]

        for *xyxy, conf, cls in results.xyxy[0]:
            label = self.model.names[int(cls)]
            if label != self.current_target:
                continue

            x1, y1, x2, y2 = map(int, xyxy)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            # 카메라 기준: 좌우 중심 offset → 몇 픽셀
            center_offset_x = cx - img_w // 2

            # 카메라 화각 설정 (ex. 60도)
            camera_fov_deg = 60.0
            degrees_per_pixel = camera_fov_deg / img_w
            pixel_angle_deg = center_offset_x * degrees_per_pixel   

            # 카메라는 전방 기준, 라이다는 후방 → 180도 보정
            lidar_angle_deg = (pixel_angle_deg + 180) % 360
            lidar_index = int(lidar_angle_deg)
            print('라이다 인덱스 ====', lidar_index)
            for start in range(0, 360, 10):
                end = start + 10
                segment = self.ranges[start:end]
                rounded = [round(val, 2) for val in segment]
                print(rounded)
            # 인덱스 유효성 체크
            if 0 <= lidar_index < len(self.lidar_data):
                distance = self.lidar_data[lidar_index]

                # distance 값이 유효한지 필터링
                if not np.isinf(distance) and not np.isnan(distance) and distance > 0.05:
                    print(f"[탐지 거리] {label} 까지 {distance:.2f}m (인덱스: {lidar_index})")
                else:
                    print(f"[무효 거리] distance={distance}, index={lidar_index}")

                if distance > 0.05:
                    if time.time() - self.last_command_time > self.command_interval:
                        self.send_hand_control_pickup()
                        self.last_command_time = time.time()
                        self.update_order_state_to_in_progress(self.current_order_id)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label.upper()} {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                    print("탐지 성공")
                    break

        cv2.imshow("Equipment Detection", frame)
        cv2.waitKey(1)
        self.is_processing = False

    def send_hand_control_pickup(self):
        msg = HandControl()
        msg.control_mode = 2
        # self.hand_control_pub.publish(msg)
        # self.get_logger().info(f"{self.current_target} 감지됨 → 픽업 명령 전송")

    def update_order_state_to_in_progress(self, order_id):
        return
        try:
            conn = pymysql.connect(**config.MYSQL, cursorclass=pymysql.cursors.DictCursor)
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

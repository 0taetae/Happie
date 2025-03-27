import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from ssafy_msgs.msg import HandControl
import torch
import cv2
import numpy as np
import time

class EquipmentDetectionNode(Node):
    def __init__(self):
        super().__init__('equipment_detection_node')

        # 이미지 구독
        self.subscriber = self.create_subscription(
            CompressedImage,
            '/image_jpeg/compressed',
            self.image_callback,
            10
        )

        # 핸드 제어 퍼블리셔
        self.hand_control_pub = self.create_publisher(
            HandControl,
            '/hand_control',
            10
        )

        # 모델 로딩
        model_path = r'C:\Users\SSAFY\Desktop\project\S12P21E103\yolov5\runs\train\exp8\weights\best.pt'
        yolov5_dir = r'C:\Users\SSAFY\Desktop\project\S12P21E103\yolov5'

        self.model = torch.hub.load(
            yolov5_dir,
            'custom',
            path=model_path,
            source='local'
        )
        self.model.conf = 0.5

        self.last_command_time = 0
        self.command_interval = 5.0  # 초 단위 (중복 명령 방지)

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        results = self.model(frame)

        object_detected = False

        for *xyxy, conf, cls in results.xyxy[0]:
            label = self.model.names[int(cls)]
            if label in ['intravenous', 'wheelchair']:
                object_detected = True

                color = (0, 255, 0) if label == 'intravenous' else (255, 0, 0)
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                              (int(xyxy[2]), int(xyxy[3])), color, 2)
                cv2.putText(frame, f'{label.upper()} {conf:.2f}',
                            (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # 감지되었고, 최근 명령 이후 일정 시간이 지났다면 제어 메시지 전송
        if object_detected and (time.time() - self.last_command_time > self.command_interval):
            self.send_hand_control_pickup()
            self.last_command_time = time.time()

        cv2.imshow("Equipment Detection", frame)
        cv2.waitKey(1)

    def send_hand_control_pickup(self):
        msg = HandControl()
        msg.control_mode = 2  # pick up
        self.hand_control_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = EquipmentDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

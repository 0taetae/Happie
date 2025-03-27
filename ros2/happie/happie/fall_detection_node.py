import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import torch
import cv2
import numpy as np
# import paho.mqtt.client as mqtt
# import json
# import base64

class FallDetectionNode(Node):
    def __init__(self):
        super().__init__('fall_detection_node')

        # 이미지 구독
        self.subscriber = self.create_subscription(
            CompressedImage,
            '/image_jpeg/compressed',
            self.image_callback,
            10
        )

        # YOLO 모델 로드
        model_path = r'C:\Users\SSAFY\Desktop\project\mobility-smarthome-skeleton\yolov5\runs\train\new_train2\weights\best.pt'
        yolov5_dir = r'C:\Users\SSAFY\Desktop\project\mobility-smarthome-skeleton\yolov5'

        self.model = torch.hub.load(
            yolov5_dir,
            'custom',
            path=model_path,
            source='local'
        )
        self.model.conf = 0.5       # 임계값

    def image_callback(self, msg):
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

        # 결과 이미지 띄우기
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

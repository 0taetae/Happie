import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import time
from sensor_msgs.msg import CompressedImage

class IMGParser(Node):
    def __init__(self):
        super().__init__('image_convertor')
        
        # 이미지 처리 속도 조절
        self.last_time = time.time()  # 마지막 이미지 처리 시간
        self.frame_skip = 5  # 5프레임마다 한 번 처리
        self.frame_count = 0

        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_jpeg/compressed',
            self.img_callback,
            10)
    
    def img_callback(self, msg):
        current_time = time.time()

        # 0.1초(10Hz) 이상 경과한 경우에만 처리
        if current_time - self.last_time < 0.1:
            return
        
        # 프레임 스킵 적용 (5프레임마다 1번 실행)
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return
        
        # 이미지 디코딩
        np_arr = np.frombuffer(msg.data, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            self.get_logger().warn("Failed to decode image")
            return
        
        # 이미지 색 변환 및 크기 조정
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_resize = cv2.resize(img_gray, (640, 480))

        # 이미지 출력
        cv2.imshow("Original Image", img_bgr)
        cv2.imshow("Gray Image", img_gray)
        cv2.imshow("Resized Image", img_resize)
        cv2.waitKey(1)

        # 마지막 처리 시간 갱신
        self.last_time = current_time

def main(args=None):
    rclpy.init(args=args)
    image_parser = IMGParser()
    rclpy.spin(image_parser)
    cv2.destroyAllWindows()  # 노드 종료 시 창 닫기

if __name__ == '__main__':
    main()


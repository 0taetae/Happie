import numpy as np
import cv2
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool


# segmentation image object detection node의 전체 로직 순서
# 1. 노드에 필요한 publisher, subscriber, timer 정의
# 2. compressed image 받기
# 3. bgr 이미지의 binarization
# 4. 물체의 contour 찾기
# 5. 물체의 bounding box 좌표 찾기
# 6. 물체의 bounding box 가 그려진 이미지 show


class IMGParser(Node):

    def __init__(self):
        super().__init__(node_name='image_parser')

        # 로직 1. 노드에 필요한 publisher, subscriber, timer 정의
        self.subscription = self.create_subscription(CompressedImage,'/image_jpeg/compressed',self.img_callback,10)
        self.object_detected_pub = self.create_publisher(Bool, '/object_detected', 1)

        #초기 이미지를 None으로 초기화한다
        self.img_bgr = None

        self.timer_period = 0.03

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.min_box_size = 30


    def img_callback(self, msg):

        # 로직 2. compressed image 받기
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


    def find_bbox(self):

        object_detected = False

        # 로직 3. bgr 이미지의 binarization
        # 지갑, 키 등의 물체에 대한 bgr 값을 알고, 이 값 범위에 해당되는
        # cv2.inRange 함수를 써서 각 물체에 대해 binarization 하십시오.

        # HSV 색상 범위 정의 (예시)
        lower_wal = np.array([0, 50, 50])  # 지갑 색상 범위 (HSV)
        upper_wal = np.array([10, 255, 255])

        lower_bp = np.array([110, 50, 50])  # 백팩 색상 범위 (HSV)
        upper_bp = np.array([130, 255, 255])

        lower_rc = np.array([50, 50, 50])  # 리모컨 색상 범위 (HSV)
        upper_rc = np.array([70, 255, 255])

        lower_key = np.array([20, 50, 50])  # 열쇠 색상 범위 (HSV)
        upper_key = np.array([40, 255, 255])

        #lower_wheelchair = np.array([97, 50, 74]) # 휠체어 색상 범위 
        #upper_wheelchair = np.array([117, 99, 154])

        # BGR에서 HSV로 변환
        img_hsv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HSV)

        # 이진화
        self.img_wal = cv2.inRange(img_hsv, lower_wal, upper_wal)
        self.img_bp = cv2.inRange(img_hsv, lower_bp, upper_bp)
        self.img_rc = cv2.inRange(img_hsv, lower_rc, upper_rc)
        self.img_key = cv2.inRange(img_hsv, lower_key, upper_key)
        #self.img_wheelchair = cv2.inRange(img_hsv, lower_wheelchair, upper_wheelchair)

        # 로직 4. 물체의 contour 찾기
        # 지갑, 키 등의 물체들이 차지한 픽셀만 흰색으로 이진화되어 있는 이미지에 대해서, 흰색 영역을 감싸는 contour들을 구하기
        contours_wal, _ = cv2.findContours(self.img_wal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_bp, _ = cv2.findContours(self.img_bp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_rc, _ = cv2.findContours(self.img_rc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_key, _ = cv2.findContours(self.img_key, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contours_wheelchair, _ = cv2.findContours(self.img_wheelchair, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ✅ 만약 bounding box가 하나라도 있으면 장애물 감지됨
        if (self.find_cnt(contours_wal) or 
            self.find_cnt(contours_bp) or 
            self.find_cnt(contours_rc) or 
            self.find_cnt(contours_key)):
            object_detected = True
        #if contours_wal or contours_bp or contours_rc or contours_key:
        #    human_detected = True
        
        # 로직 5. 물체의 bounding box 좌표 찾기
        
        #self.find_cnt(contours_wal)
        
        #self.find_cnt(contours_bp)
        
        #self.find_cnt(contours_rc)
        
        #self.find_cnt(contours_key)

        #self.human_detected_pub.publish(Bool(data=human_detected))


    def find_cnt(self, contours):

        # 로직 5. 물체의 bounding box 좌표 찾기
        # 지갑, 키 등의 물체들의 흰색 영역을 감싸는 contour 결과를 가지고
        # bbox를 원본 이미지에 draw 하십시오.
        detected = False
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > self.min_box_size and h > self.min_box_size:  # 최소 크기 필터 적용
                cv2.rectangle(self.img_bgr, (x, y), (x + w, y + h), (0, 255, 255), 2)
                detected = True  # 일정 크기 이상의 물체가 감지되었음을 표시
        return detected


    def timer_callback(self):

        if self.img_bgr is not None:
            
            # 이미지가 ros subscriber로 받고 None이 아닌 array로 정의됐을 때,
            # object에 대한 bbox 추정을 시작.             
            self.find_bbox()

            # 로직 6. 물체의 bounding box 가 그려진 이미지 show
            cv2.imshow("seg_results", self.img_bgr)

            cv2.waitKey(1)
        else:
            pass


def main(args=None):

    rclpy.init(args=args)

    image_parser = IMGParser()

    rclpy.spin(image_parser)


if __name__ == '__main__':

    main()

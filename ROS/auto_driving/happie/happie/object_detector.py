import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage, LaserScan
from std_msgs.msg import Float32MultiArray
from ssafy_msgs.msg import BBox
from std_msgs.msg import Bool
from std_msgs.msg import Int32

# object detector node의 전체 로직 순서
# 로직 1 : 노드에 필요한 publisher, subscriber, descriptor, detector 정의
# 로직 2 : image binarization
# 로직 3 : object detection 실행 후 bounding box 출력
# 로직 4 : non maximum supression으로 bounding box 정리
# 로직 5 : bbox를 ros msg 파일에 write
# 로직 6 : bbox를 원본 이미지에 draw
# 로직 7 : bbox 결과 show
# 로직 8 : bbox msg 송신

def non_maximum_supression(bboxes, threshold=0.5):

    # 로직 1 : bounding box 크기 역순으로 sort
    bboxes = sorted(bboxes, key=lambda detections: detections[3],
                    reverse=True)
    new_bboxes = []

    # 로직 2 : new_bboxes 리스트 정의 후 첫 bbox save
    new_bboxes.append(bboxes[0])

    # 로직 3 : 기존 bbox 리스트에 첫 bbox delete
    bboxes.pop(0)

    for bbox in bboxes:
        overlap = False
        for new_bbox in new_bboxes:
            x1_tl = bbox[0]
            x2_tl = new_bbox[0]
            x1_br = bbox[0] + bbox[2]
            x2_br = new_bbox[0] + new_bbox[2]
            y1_tl = bbox[1]
            y2_tl = new_bbox[1]
            y1_br = bbox[1] + bbox[3]
            y2_br = new_bbox[1] + new_bbox[3]

            # 로직 4 : 두 bbox의 겹치는 영역을 구해서, 영역이 안 겹칠때 new_bbox로 save
            x_overlap = max(0, min(x1_br, x2_br) - max(x1_tl, x2_tl))
            y_overlap = max(0, min(y1_br, y2_br) - max(y1_tl, y2_tl))
            overlap_area = x_overlap * y_overlap

            area_1 = bbox[2] * bbox[3]
            area_2 = new_bbox[2] * new_bbox[3]

            total_area = area_1 + area_2 - overlap_area
            overlap_ratio = overlap_area / float(total_area)

            if overlap_ratio < threshold:
                new_bboxes.append(bbox)
                break

            if overlap_ratio >= threshold:
                overlap = True

        if not overlap:
            new_bboxes.append(bbox)

    return new_bboxes


class ObjectDetector(Node):

    def __init__(self):
        super().__init__(node_name='object_detector')

        # 로직 1 : 노드에 필요한 publisher, subscriber, descriptor, detector, timer 정의
        self.subs_lidar = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 1)
        self.subs_img = self.create_subscription(CompressedImage,'/image_jpeg/compressed',self.img_callback,1)
        self.object_detected_pub = self.create_publisher(Int32, '/object_detected', 1)

        self.img_bgr = None
        self.lidar_ranges = []
        self.object_detected = False

        #self.timer_period = 0.03

        self.timer = self.create_timer(0.03, self.timer_callback)

        self.bbox_pub_ = self.create_publisher(BBox, '/bbox', 1)

        self.pedes_detector = cv2.HOGDescriptor()
        self.pedes_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.able_to_pub = True

    def lidar_callback(self, msg):
        self.lidar_ranges = np.array(msg.ranges)

    def img_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    

    # run_mapping 에서 update에 있는 laser 로직 보기(일정 거리내에, 일정 heading 내에 있는 사물 감지하도록)
    def detect_object(self, img_bgr):
        self.bbox_msg = BBox()

        # 로직 2 : image grayscale conversion
        img_pre = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 로직 3 : object detection 실행 후 bounding box 출력
        (rects_temp, _) = self.pedes_detector.detectMultiScale(img_pre, winStride=(2, 2), padding=(8, 8), scale=1.3)

        #print(self.lidar_ranges,'lidar_ranges')

        if len(rects_temp) != 0:
            # 로직 4 : non maximum supression으로 bounding box 정리
            rects = non_maximum_supression(rects_temp)

            

            # ✅ LiDAR 거리 필터링 (가까운 사람만 선택)
            close_rects = []
            for (x, y, w, h) in rects:
                bbox_center_x = x + w // 2
                bbox_center_y = y + h // 2

                # 1. 좌표를 각도로 변환
                angle_index = (int((bbox_center_x / img_bgr.shape[1]) * len(self.lidar_ranges)) +180)%360
                #print(self.lidar_ranges,'lidar_ranges')
                #print(angle_index,'angle_index')
                # 2. LiDAR 거리 확인
                if 0 <= angle_index < len(self.lidar_ranges):
                    # angle = (angle_index / len(self.lidar_ranges)) * 360
                    if not (330 < angle_index or angle_index < 30):
                        continue

                    distance = self.lidar_ranges[angle_index]
                    #print(distance,'거리')
                    
                    if distance < 2: 
                        close_rects.append((x, y, w, h))
                        #print(distance,'거리 내')
                        if not self.object_detected:  # 이미 감지된 상태라면 메시지를 보내지 않음
                            self.object_detected_pub.publish(Int32(data=angle_index))
                            self.object_detected = True  # 감지 상태 유지
                            print("========사람 감지 - 메시지 송신=========")
                            print(f"감지 각도: {angle_index:.2f}° / 거리: {distance:.2f}m")
                    

            # if close_rects and not self.object_detected:
            #     self.object_detected_pub.publish(Bool(data=True))
            #     self.object_detected = True
            #     print("======== 가까운 사람 감지 - 메시지 송신 =========")

            # 로직 5 : bbox를 ros msg 파일에 write
            xl, yl, wl, hl = [], [], [], []
            for (x, y, w, h) in rects:
                xl.append(int(x))
                yl.append(int(y))
                wl.append(int(w))
                hl.append(int(h))

            # if self.able_to_pub:
            #     self.bbox_msg.num_bbox = len(rects)
            #     self.bbox_msg.idx_bbox = list(range(len(rects)))  # idx_bbox에 정수 식별자 할당
            #     self.bbox_msg.x = xl
            #     self.bbox_msg.y = yl
            #     self.bbox_msg.w = wl
            #     self.bbox_msg.h = hl



            #self.object_detected_pub.publish(Bool(data=True))
            #print("사람 감지")

            # 로직 6 : bbox를 원본 이미지에 draw
            # for (x, y, w, h) in rects:
            #     cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 255), 2)

            self.bbox_msg.num_bbox = len(close_rects)
            self.bbox_msg.idx_bbox = list(range(len(close_rects)))
            self.bbox_msg.x = xl
            self.bbox_msg.y = yl
            self.bbox_msg.w = wl
            self.bbox_msg.h = hl

            for (x, y, w, h) in close_rects:
                cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 255), 2)


        else:
            # self.bbox_msg.num_bbox = len(rects_temp)
            # self.object_detected_pub.publish(Bool(data=False))
            self.bbox_msg.num_bbox = 0
            if self.object_detected:  
                print("사람 사라짐 - 감지 상태 초기화")  
                self.object_detected = False  # 사람이 사라지면 다시 감지할 수 있도록 상태 초기화
                #self.object_detected_pub.publish(Bool(data=False))

        # 로직 7 : bbox 결과 show
        cv2.imshow("detection result", img_bgr)
        cv2.waitKey(1)

    def timer_callback(self):
        if self.img_bgr is not None:
            self.detect_object(self.img_bgr)

            # 로직 8 : bbox msg 송신
            self.bbox_pub_.publish(self.bbox_msg)

        else:
            pass


def main(args=None):
    rclpy.init(args=args)
    hdetector = ObjectDetector()
    rclpy.spin(hdetector)


if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
import ros2pkg
from geometry_msgs.msg import Twist, PoseStamped, Pose, TransformStamped
from ssafy_msgs.msg import TurtlebotStatus
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Float32
from squaternion import Quaternion
from nav_msgs.msg import Odometry, Path, OccupancyGrid, MapMetaData
from math import pi, cos, sin, sqrt
import tf2_ros
import heapq
import os
import happie.utils as utils
import numpy as np
import cv2
import time

from .config import params_map, PKG_PATH, MQTT_CONFIG
import paho.mqtt.client as mqtt

import matplotlib.pyplot as plt
from std_msgs.msg import Bool
from sensor_msgs.msg import CompressedImage

from PIL import Image 
import io
import base64  

# mapping node의 전체 로직 순서
# 1. publisher, subscriber, msg 생성
# 2. mapping 클래스 생성
# 3. 맵의 resolution, 중심좌표, occupancy에 대한 threshold 등의 설정 받기
# 4. laser scan 메시지 안의 ground truth pose 받기
# 5. lidar scan 결과 수신
# 6. map 업데이트 시작
# 7. pose 값을 받아서 좌표변환 행렬로 정의
# 8. laser scan 데이터 좌표 변환
# 9. pose와 laser의 grid map index 변환
# 10. laser scan 공간을 맵에 표시
# 11. 업데이트 중인 map publish
# 12. 맵 저장

## Bresenham's Algorithm
def createLineIterator(P1, P2, img):
    # Bresenham's line algorithm을 구현해서 이미지에 직선을 그리는 메소드

    imageH, imageW = img.shape[:2]
    P1X, P1Y = P1
    P2X, P2Y = P2
 
    ## 로직 1 : 두 점을 있는 백터의 x, y 값과 크기 계산
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = abs(dX)
    dYa = abs(dY)
    
    ## 로직 2 : 직선을 그릴 grid map의 픽셀 좌표를 넣을 numpy array 를 predifine
    itbuffer = np.empty((max(dYa, dXa), 2), dtype=np.int32)
    itbuffer.fill(np.nan)

    ## 로직 3 : 직선 방향 체크
    negY = P1Y > P2Y
    negX = P1X > P2X
    
    ## 로직 4 : 수직선의 픽셀 좌표 계산   
    if P1X == P2X:
        itbuffer[:, 0] = P1X
        itbuffer[:, 1] = np.arange(P1Y, P2Y, -1 if negY else 1)
    ## 로직 5 : 수평선의 픽셀 좌표 계산
    elif P1Y == P2Y:
        itbuffer[:, 1] = P1Y
        itbuffer[:, 0] = np.arange(P1X, P2X, -1 if negX else 1)
    ## 로직 6 : 대각선의 픽셀 좌표 계산  
    else:
        steepSlope = dYa > dXa
        slope = dY / dX
        if steepSlope:
            itbuffer[:, 1] = np.arange(P1Y, P2Y, -1 if negY else 1)
            itbuffer[:, 0] = P1X + (itbuffer[:, 1] - P1Y) / slope
        else:
            itbuffer[:, 0] = np.arange(P1X, P2X, -1 if negX else 1)
            itbuffer[:, 1] = P1Y + (itbuffer[:, 0] - P1X) * slope
    
    itbuffer = itbuffer[(itbuffer[:, 0] >= 0) & (itbuffer[:, 0] < imageW) & (itbuffer[:, 1] >= 0) & (itbuffer[:, 1] < imageH)]
    return itbuffer


class Mapping:
    # 사용자가 정의한 맵 설정을 받아서 회색의 어레이로 초기화 시키고,
    # 로봇의 pose와 2d 라이다 값들을 받은 다음,
    # 라이다가 나타내는 로봇으로부터 측정된 좌표와의 직선을
    # utils_skeleton.py에 있는 createLineIterator()로
    # 그려가면서 맵을 채워서 저장할 수 있도록 만든 스크립트입니다.

    def __init__(self, params_map):
        # 로직 3. 맵의 resolution, 중심좌표, occupancy에 대한 threshold 등의 설정들을 받습니다
        self.map_resolution = params_map["MAP_RESOLUTION"]
        self.map_size = np.array(params_map["MAP_SIZE"]) / self.map_resolution
        self.map_size = self.map_size.astype(int)
        self.map_center = params_map["MAP_CENTER"]
        self.map = np.ones((self.map_size[0].astype(np.int), self.map_size[1].astype(np.int)))*0.5
        self.occu_up = params_map["OCCUPANCY_UP"]
        self.occu_down = params_map["OCCUPANCY_DOWN"]

        self.map_filename = params_map["MAP_FILENAME"]
        self.map_vis_resize_scale = params_map["MAPVIS_RESIZE_SCALE"]

        self.T_r_l = np.array([[0,-1,0],[1,0,0],[0,0,1]])
        # 🔥 기존 맵 파일이 있으면 로드
        map_path = os.path.join(PKG_PATH, '..', 'data', 'update_map.txt')
        if os.path.exists(map_path):
            print(f"기존 맵 {map_path} 불러오기...")
            
            print(self.map_size[0])
            print(self.map_size[1])
            with open(map_path, 'r') as f:
                existing_data = list(map(float, f.read().split()))
                print(len(existing_data))

            if len(existing_data) == self.map_size[0] * self.map_size[1]:
                self.map = np.array(existing_data).reshape(self.map_size[0], self.map_size[1])
            else:
                print("⚠ 기존 맵 크기가 현재 설정과 다름 → 새 맵 생성")
                self.map = np.ones((self.map_size[0].astype(int), self.map_size[1].astype(int))) * 0.5
        else:
            print("📂 기존 맵 없음 → 새 맵 생성")
            self.map = np.ones((self.map_size[0].astype(int), self.map_size[1].astype(int))) * 0.5


    def update(self, pose, laser):
        # print("update start!!!")
        # 로직 7. pose 값을 받아서 좌표변환 행렬로 정의
        n_points = laser.shape[1]
        pose_mat = utils.xyh2mat2D(pose)


        # 로직 8. laser scan 데이터 좌표 변환
        pose_mat = np.matmul(pose_mat,self.T_r_l)
        laser_mat = np.ones((3, n_points))
        laser_mat[:2, :] = laser

        laser_global = np.matmul(pose_mat, laser_mat)
        
        # 로직 9. pose와 laser의 grid map index 변환
        pose_x = (pose[0] - self.map_center[0] + (self.map_size[0]*self.map_resolution)/2) / self.map_resolution
        pose_y = (pose[1] - self.map_center[1] + (self.map_size[1]*self.map_resolution)/2) / self.map_resolution
        laser_global_x = (laser_global[0, :] - self.map_center[0] + (self.map_size[0]*self.map_resolution)/2) / self.map_resolution
        laser_global_y = (laser_global[1, :] - self.map_center[1] + (self.map_size[1]*self.map_resolution)/2) / self.map_resolution

        
        # 로직 10. laser scan 공간을 맵에 표시
        for i in range(laser_global.shape[1]):
            p1 = np.array([pose_x, pose_y]).reshape(-1).astype(np.int32)
            p2 = np.array([laser_global_x[i], laser_global_y[i]]).astype(np.int32)

            line_iter = createLineIterator(p1, p2, self.map)
            # print(line_iter)

            if line_iter.shape[0] == 0:
                continue

            avail_x = line_iter[:, 0].astype(np.int32)
            avail_y = line_iter[:, 1].astype(np.int32)

            ## Empty
            self.map[avail_y[:-1], avail_x[:-1]] -= self.occu_down
            self.map[avail_y[:-1], avail_x[:-1]] = np.clip(self.map[avail_y[:-1], avail_x[:-1]], 0, 1)

            ## Occupied
            self.map[avail_y[-1], avail_x[-1]] += self.occu_up
            self.map[avail_y[-1], avail_x[-1]] = np.clip(self.map[avail_y[-1], avail_x[-1]], 0, 1)
                
        self.show_pose_and_points(pose, laser_global) 
        cv2.waitKey(1)

    def __del__(self):

        pass

    def show_pose_and_points(self, pose, laser_global):
        # run_mapping 실행 시, 성능이슈로 시각화x
        
        tmp_map = self.map.astype(np.float32)
        map_bgr = cv2.cvtColor(tmp_map, cv2.COLOR_GRAY2BGR)

        pose_x = (pose[0] - self.map_center[0] + (self.map_size[0]*self.map_resolution)/2) / self.map_resolution
        pose_y = (pose[1] - self.map_center[1] + (self.map_size[1]*self.map_resolution)/2) / self.map_resolution

        laser_global_x = (laser_global[0, :] - self.map_center[0] + (self.map_size[0]*self.map_resolution)/2) / self.map_resolution
        laser_global_y =  (laser_global[1, :] - self.map_center[1] + (self.map_size[1]*self.map_resolution)/2) / self.map_resolution

        for i in range(laser_global.shape[1]):
            (l_x, l_y) = np.array([laser_global_x[i], laser_global_y[i]]).astype(np.int)
            center = (l_x, l_y)
            cv2.circle(map_bgr, center, 1, (0,255,0), -1)

        center = (pose_x.astype(np.int32)[0], pose_y.astype(np.int32)[0])
        
        cv2.circle(map_bgr, center, 2, (0,0,255), -1)

        map_bgr = cv2.resize(map_bgr, dsize=(0, 0), fx=self.map_vis_resize_scale, fy=self.map_vis_resize_scale)
        # print("Map shape:", map_bgr.shape)
        cv2.imshow('Sample Map', map_bgr)
        cv2.waitKey(1)



class Mapper(Node):
    print("Mapper start!!!")
    def __init__(self):
        super().__init__('Mapper')
        self.last_save_time = time.time()
        
        # 로직 1 : publisher, subscriber, msg 생성
        self.subscription = self.create_subscription(LaserScan,'/scan',self.scan_callback,10)
        # self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.image_sub = self.create_subscription(CompressedImage,'/image_jpeg/compressed',self.image_callback,1)

        # MQTT 설정
        self.mqtt_client = mqtt.Client()
        self.mqtt_broker = MQTT_CONFIG["BROKER"]
        self.mqtt_port = MQTT_CONFIG["PORT"]
        self.mqtt_topic_position = "robot/map_position"
        self.mqtt_topic_image = "robot/image"
        #self.mqtt_topic_destination = "robot/destination"

        self.mqtt_client.username_pw_set(MQTT_CONFIG["USERNAME"], MQTT_CONFIG["PASSWORD"])
        #self.mqtt_client.loop_start()
        # MQTT 브로커에 연결
        self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
        self.mqtt_client.loop_start()  # 비동기 처리 시작

        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 1)
        
        self.map_msg=OccupancyGrid()
        self.map_msg.header.frame_id="map"
        self.map_size=int(params_map["MAP_SIZE"][0]\
            /params_map["MAP_RESOLUTION"]*params_map["MAP_SIZE"][1]/params_map["MAP_RESOLUTION"])
        
                
        self.map_pose_x = 0
        self.map_pose_y = 0

        #self.destination_x = 0.0
        #self.destination_y = 0.0

        m = MapMetaData()
        m.resolution = params_map["MAP_RESOLUTION"]
        m.width = int(params_map["MAP_SIZE"][0]/params_map["MAP_RESOLUTION"])
        m.height = int(params_map["MAP_SIZE"][1]/params_map["MAP_RESOLUTION"])
        quat = np.array([0, 0, 0, 1])
        m.origin = Pose()

        m.origin.position.x = ((params_map["MAP_CENTER"][0]-params_map["MAP_SIZE"][0])/2)
        m.origin.position.y = ((params_map["MAP_CENTER"][1]-params_map["MAP_SIZE"][0])/2)

        self.map_meta_data = m

        self.map_msg.info=self.map_meta_data

        # 로직 2 : mapping 클래스 생성
        self.mapping = Mapping(params_map)

    def image_callback(self, msg):
        print('이미지 콜백 시작!!!!!')
        try:
            # 1. ROS 이미지 데이터를 numpy 배열로 변환
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # OpenCV 이미지로 디코딩

            # 2. 다운스케일 처리 (예: 50% 크기로 축소)
            scale_percent = 50  # 축소 비율 (%)
            width = int(cv_image.shape[1] * scale_percent / 100)
            height = int(cv_image.shape[0] * scale_percent / 100)
            resized_image = cv2.resize(cv_image, (width, height), interpolation=cv2.INTER_AREA)

            # 3. 이미지를 PIL로 변환하여 base64 인코딩
            pil_image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG')  # JPEG로 저장
            encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # 4. MQTT로 전송
            self.mqtt_client.publish(self.mqtt_topic_image, encoded_image)
            print("다운스케일된 이미지 MQTT 전송 완료")

        except Exception as e:
            print(f"이미지 MQTT 전송 실패: {e}")
    
    def scan_callback(self, msg):
        print("scan_callback start!!!")
    
        # [1] 현재 위치 (pose_x, pose_y, heading) 가져오기
        pose_x = msg.range_min  # 실제 x 좌표 (meters)
        pose_y = msg.scan_time  # 실제 y 좌표 (meters)
        heading = msg.time_increment  # 로봇의 방향 (radians)
        # print(pose_x,pose_y,'실제 위치')
        
        # [2] 거리 데이터를 기반으로 LIDAR 스캔 변환
        distance = np.array(msg.ranges)
        angles = np.linspace(0, 2 * np.pi, len(distance), endpoint=False)
        x = distance * np.cos(angles)
        y = distance * np.sin(angles)
        laser = np.vstack((x, y))  
    
        # [3] 현재 위치를 Grid Map 좌표계로 변환
        MAP_RESOLUTION = params_map["MAP_RESOLUTION"]
        MAP_CENTER = params_map["MAP_CENTER"]
        MAP_SIZE = params_map["MAP_SIZE"]
    
        map_x = (pose_x - MAP_CENTER[0] + MAP_SIZE[0]/2) / MAP_RESOLUTION
        map_y = (pose_y - MAP_CENTER[1] + MAP_SIZE[1]/2) / MAP_RESOLUTION
        self.map_pose_x = map_x
        self.map_pose_y = map_y
    
        # 맵 퍼블리시
        # 각도 계산 (1도씩 증가하므로, 각도를 라디안으로 변환)
        angles = np.linspace(0, 2 * np.pi, len(distance), endpoint=False)  # 360개의 각도 생성 (0에서 2π까지)

        # 거리 데이터를 기반으로 x, y 좌표 계산
        x = distance * np.cos(angles)  # x = 거리 * cos(각도)
        y = distance * np.sin(angles)  # y = 거리 * sin(각도)

        laser = np.vstack((x, y))  # x, y 값을 결합하여 레이저 좌표를 생성

        # 로봇의 현재 위치 
        map_x = (pose_x - params_map["MAP_CENTER"][0] + params_map["MAP_SIZE"][0]/2) / params_map["MAP_RESOLUTION"]
        map_y = (pose_y - params_map["MAP_CENTER"][1] + params_map["MAP_SIZE"][1]/2) / params_map["MAP_RESOLUTION"]

        # MQTT로 위치 데이터 전송
        mqtt_payload = f"{map_x:.0f},{map_y:.0f}"
        try:
            self.mqtt_client.publish(self.mqtt_topic_position, mqtt_payload)
            print(f"MQTT 발행(위치 데이터): {mqtt_payload}")
        except Exception as e:
            print(f"MQTT 발행 실패: {e}")

        # map 업데이트 실행
        pose = np.array([[pose_x], [pose_y], [heading]])
        self.mapping.update(pose, laser)

        # 로그 출력 (현재 위치 확인)
        #print(f"현재 위치 (실제 좌표): x={pose_x:.2f}, y={pose_y:.2f}, heading={heading:.2f} rad")
        #print(f"맵 좌표계 인덱스: map_x={map_x:.0f}, map_y={map_y:.0f}")
        
        np_map_data = self.mapping.map.reshape(-1)
        list_map_data = [100 - int(value * 100) for value in np_map_data]
        list_map_data = [max(0, min(100, v)) for v in list_map_data]
    
        self.map_msg.header.stamp = rclpy.clock.Clock().now().to_msg()
        self.map_msg.data = np.clip((self.mapping.map.flatten() * 100), -128, 127).astype(np.int32).tolist()
        self.map_pub.publish(self.map_msg)
    
        # 3초마다 맵 저장
        current_time = time.time()
        if current_time - self.last_save_time > 3:
            save_map(self, 'update_map.txt')
            self.last_save_time = current_time


def save_map(node, file_path):
    print("save map start!!!")
    
    # 맵 저장
    pkg_path = PKG_PATH
    back_folder='..'
    folder_name='data'
    file_name=file_path
    full_path=os.path.join(pkg_path,back_folder,folder_name,file_name)
    print(full_path)
    
    f=open(full_path,'w')
    data=''
    for pixel in node.map_msg.data :
        data+='{0} '.format(pixel)
    print("map 데이터 저장 완료")
    f.write(data) 
    f.close()


def main(args=None):    
    rclpy.init(args=args)
    
    try :    
        run_mapping = Mapper()
        rclpy.spin(run_mapping)
    except Exception as e:
        print(f"Error: {e}")
        # save_map(run_mapping, 'map.txt')
    finally:
        if 'run_mapping' in locals():
        #     print('최종 map 저장')
        #     save_map(run_mapping, 'map.txt')
            run_mapping.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
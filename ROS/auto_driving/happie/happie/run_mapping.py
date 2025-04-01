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
import os
import happie.utils as utils
import numpy as np
import cv2
import time

from .config import params_map, PKG_PATH

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
        self.map_center = params_map["MAP_CENTER"]
        self.map = np.ones((self.map_size[0].astype(np.int), self.map_size[1].astype(np.int)))*0.5
        self.occu_up = params_map["OCCUPANCY_UP"]
        self.occu_down = params_map["OCCUPANCY_DOWN"]

        self.map_filename = params_map["MAP_FILENAME"]
        self.map_vis_resize_scale = params_map["MAPVIS_RESIZE_SCALE"]

        self.T_r_l = np.array([[0,-1,0],[1,0,0],[0,0,1]])


    def update(self, pose, laser):
        print("update start!!!")
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

    def __del__(self):
        # 로직 12. 종료 시 map 저장
        ## Ros2의 노드가 종료될 때 만들어진 맵을 저장하도록 def __del__과 save_map이 정의되어 있습니다
        ## self.save_map(())
        pass

    
    ## def save_map(self):
    ##    map_clone = self.map.copy()
    ##    cv2.imwrite(self.map_filename, map_clone*255)



    def show_pose_and_points(self, pose, laser_global):
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
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 1)
        
        self.map_msg=OccupancyGrid()
        self.map_msg.header.frame_id="map"
        self.map_size=int(params_map["MAP_SIZE"][0]\
            /params_map["MAP_RESOLUTION"]*params_map["MAP_SIZE"][1]/params_map["MAP_RESOLUTION"])
        

        m = MapMetaData()
        m.resolution = params_map["MAP_RESOLUTION"]
        m.width = int(params_map["MAP_SIZE"][0]/params_map["MAP_RESOLUTION"])
        m.height = int(params_map["MAP_SIZE"][1]/params_map["MAP_RESOLUTION"])
        quat = np.array([0, 0, 0, 1])
        m.origin = Pose()
        m.origin.position.x = 375.0
        m.origin.position.y = 375.0
        
        print(m.origin.position.x, '=====')
        print(m.origin.position.y, '=====')
        self.map_meta_data = m

        self.map_msg.info=self.map_meta_data

        # 로직 2 : mapping 클래스 생성
        self.mapping = Mapping(params_map)

    
    def scan_callback(self, msg):
        print("scan_callback start!!!")
        # 로직 4 : laser scan 메시지 안의 ground truth pose 받기
        pose_x = msg.range_min
        pose_y = msg.scan_time
        heading = msg.time_increment
        
        # 로직 5 : lidar scan 결과 수신
        distance = np.array(msg.ranges)  # LaserScan의 거리 데이터 (길이 360의 배열)

        # 각도 계산 (1도씩 증가하므로, 각도를 라디안으로 변환)
        angles = np.linspace(0, 2 * np.pi, len(distance), endpoint=False)  # 360개의 각도 생성 (0에서 2π까지)

        # 거리 데이터를 기반으로 x, y 좌표 계산
        x = distance * np.cos(angles)  # x = 거리 * cos(각도)
        y = distance * np.sin(angles)  # y = 거리 * sin(각도)

        laser = np.vstack((x, y))  # x, y 값을 결합하여 레이저 좌표를 생성

        # 로직 6 : map 업데이트 실행(4,5번이 완성되면 바로 주석처리된 것을 해제하고 쓰시면 됩니다.)
        pose = np.array([[pose_x], [pose_y], [heading]])
        self.mapping.update(pose, laser)
        
        np_map_data = self.mapping.map.reshape(-1)
        list_map_data = [100 - int(value * 100) for value in np_map_data]
        list_map_data = [max(0, min(100, v)) for v in list_map_data]

        # 로직 11 : 업데이트 중인 map publish

        self.map_msg.header.stamp =rclpy.clock.Clock().now().to_msg()
        self.map_msg.data = (self.mapping.map.flatten() * 100).astype(np.int32).tolist()
        self.map_pub.publish(self.map_msg)

        current_time = time.time()
        if current_time - self.last_save_time > 10:  # 10초마다 저장
            save_map(self, 'map.txt')
            self.last_save_time = current_time



def save_map(node, file_path):
    print("save map start!!!")
    
    # 로직 12 : 맵 저장
    pkg_path = PKG_PATH
    back_folder = '..'
    folder_name = 'data'
    file_name = file_path
    full_path = os.path.join(pkg_path, back_folder, folder_name, file_name)
    print(full_path)
    
    # node.map_msg.data가 1D 배열이므로 2D 배열로 변환 (예: 맵 크기 지정)
    map_width = params_map['MAP_SIZE'][0]/params_map['MAP_RESOLUTION']
    map_height = params_map['MAP_SIZE'][1]/params_map['MAP_RESOLUTION']
    
    # 1D 데이터를 2D 배열로 변환
    map_data = np.array(node.map_msg.data).reshape(map_height, map_width)

    # 회전된 맵을 1D 배열로 다시 변환
    map_data_flat = map_data.flatten()

    # 파일에 저장
    with open(full_path, 'w') as f:
        data = ''
        for pixel in map_data_flat:
            data += '{0} '.format(pixel)
        print("map 데이터 저장 완료")
        f.write(data)


def main(args=None):    
    rclpy.init(args=args)
    
    try :    
        run_mapping = Mapper()
        rclpy.spin(run_mapping)
    except Exception as e:
        print(f"Error: {e}")
        save_map(run_mapping, 'map.txt')
    finally:
        if 'run_mapping' in locals():
        #     print('최종 map 저장')
        #     save_map(run_mapping, 'map.txt')
            run_mapping.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
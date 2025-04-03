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

# A* 경로를 ROS Path 메시지로 변환하는 함수
def convert_path_to_ros(path, map_center, map_resolution):
    ros_path = Path()
    ros_path.header.frame_id = "map"  # TF 좌표계 설정
    ros_path.header.stamp = rclpy.clock.Clock().now().to_msg()

    for node in path:
        i, j = node  # A*에서의 (grid y, grid x) 좌표

        # Grid 좌표 → 실제 좌표 변환
        real_x = map_center[0] + (j * map_resolution)
        real_y = map_center[1] + (i * map_resolution)

        # PoseStamped 메시지 생성
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = real_x
        pose.pose.position.y = real_y
        pose.pose.position.z = 0.0  # 2D 경로이므로 z=0

        ros_path.poses.append(pose)

    return ros_path

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
        cv2.waitKey(1)

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
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, 'goal_pose', self.goal_callback, 1)
        self.global_path_pub = self.create_publisher(Path, 'a_star_global_path', 10)

        # MQTT 설정
        self.mqtt_client = mqtt.Client()
        self.mqtt_broker = MQTT_CONFIG["BROKER"]
        self.mqtt_port = MQTT_CONFIG["PORT"]
        self.mqtt_topic = "robot/map_position"

        self.mqtt_client.username_pw_set(MQTT_CONFIG["USERNAME"], MQTT_CONFIG["PASSWORD"])

        # MQTT 브로커에 연결
        self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
        self.mqtt_client.loop_start()  # 비동기 처리 시작

        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 1)
        
        self.map_msg=OccupancyGrid()
        self.map_msg.header.frame_id="map"
        self.map_size=int(params_map["MAP_SIZE"][0]\
            /params_map["MAP_RESOLUTION"]*params_map["MAP_SIZE"][1]/params_map["MAP_RESOLUTION"])
        
        # 현재 위치 및 방향
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.yaw = 0.0
        
        self.grid = []
        self.rows = []
        self.cols = []
                
        self.map_pose_x = 0
        self.map_pose_y = 0

        m = MapMetaData()
        m.resolution = params_map["MAP_RESOLUTION"]
        m.width = int(params_map["MAP_SIZE"][0]/params_map["MAP_RESOLUTION"])
        m.height = int(params_map["MAP_SIZE"][1]/params_map["MAP_RESOLUTION"])
        quat = np.array([0, 0, 0, 1])
        m.origin = Pose()

        m.origin.position.x = ((params_map["MAP_CENTER"][0]-params_map["MAP_SIZE"][0])/2)
        m.origin.position.y = ((params_map["MAP_CENTER"][1]-params_map["MAP_SIZE"][0])/2)

        
        print(m.origin.position.x, '=====')
        print(m.origin.position.y, '=====')
        self.map_meta_data = m

        self.map_msg.info=self.map_meta_data

        # 로직 2 : mapping 클래스 생성
        self.mapping = Mapping(params_map)

    def heuristic(self, a, b):
        # 맨해튼 거리 (거리 계산 방법을 변경할 수 있음)
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def neighbors(self, node):
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # 상, 하, 좌, 우
            (-1, -1), (-1, 1), (1, -1), (1, 1) # 대각선
        ]
        dCost = [1, 1, 1, 1, 1.414, 1.414, 1.414, 1.414] # 이동 비용 설정
        neighbors = []
        # 경계를 벗어나지 않고 벽(40 이상)이 아니면 유효한 인접 노드
        for i, direction in enumerate(directions):
            neighbor = (node[0] + direction[0], node[1] + direction[1])
            if 0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols and self.grid[neighbor[0]][neighbor[1]] < 40:
                neighbors.append((neighbor, dCost[i]))
        return neighbors

    def a_star(self, start, goal):

        def grid_to_real(path, params):
            grid_size = int(params["MAP_SIZE"][0] / params["MAP_RESOLUTION"])  # 그리드 크기 계산
            x_center, y_center = params["MAP_CENTER"]  # 맵 중심 좌표
            resolution = params["MAP_RESOLUTION"]  # 해상도

            real_path = [
                (
                    x_center + (j - grid_size // 2) * resolution,
                    y_center + (i - grid_size // 2) * resolution
                )
                for i, j in path
            ]
            return real_path
        
        open_list = []
        closed_list = set()
        
        heapq.heappush(open_list, (0 + self.heuristic(start, goal), 0, self.heuristic(start, goal), start))
        
        came_from = {}
        g_score = {start: 0}
        
        while open_list:
            current_f, current_g, current_h, current_node = heapq.heappop(open_list)
            
            if current_node == goal:
                path = []
                while current_node in came_from:
                    path.append(current_node)
                    current_node = came_from[current_node]
                path.append(start)
                real_path = grid_to_real(path[::-1], params_map)
                return path[::-1], real_path
            
            closed_list.add(current_node)
            
            for neighbor, cost in self.neighbors(current_node):
                if neighbor in closed_list:
                    continue
                
                tentative_g_score = current_g + cost  
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score, tentative_g_score, self.heuristic(neighbor, goal), neighbor))
                    came_from[neighbor] = current_node

        return None

    def publish_global_path(self, path_points):
        """경로를 Path 메시지로 변환 후 Publish"""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"

        for x, y in path_points:
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "map"
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0  # 기본 방향 설정
            path_msg.poses.append(pose)

        self.global_path_pub.publish(path_msg)
        self.get_logger().info("Published global path.")


    def scan_callback(self, msg):
        # print("scan_callback start!!!")
    
        # [1] 현재 위치 (pose_x, pose_y, heading) 가져오기
        pose_x = msg.range_min  # 실제 x 좌표 (meters)
        pose_y = msg.scan_time  # 실제 y 좌표 (meters)
        heading = msg.time_increment  # 로봇의 방향 (radians)
        
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

        # pose = np.array([[pose_x], [pose_y], [heading]])
        # self.mapping.update(pose, laser)
    
        # [4] 로그 출력 (현재 위치 확인)
        print(f"현재 위치 (실제 좌표): x={pose_x:.2f}, y={pose_y:.2f}, heading={heading:.2f} rad")
        print(f"맵 좌표계 인덱스: map_x={map_x:.0f}, map_y={map_y:.0f}")
    
        # [5] 맵 퍼블리시
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
            self.mqtt_client.publish(self.mqtt_topic, mqtt_payload)
            print(f"MQTT 발행: {mqtt_payload}")
        except Exception as e:
            print(f"MQTT 발행 실패: {e}")

        # 로직 6 : map 업데이트 실행
        pose = np.array([[pose_x], [pose_y], [heading]])
        self.mapping.update(pose, laser)

        # [4] 로그 출력 (현재 위치 확인)
        print(f"현재 위치 (실제 좌표): x={pose_x:.2f}, y={pose_y:.2f}, heading={heading:.2f} rad")
        print(f"맵 좌표계 인덱스: map_x={map_x:.0f}, map_y={map_y:.0f}")
        
        np_map_data = self.mapping.map.reshape(-1)
        list_map_data = [100 - int(value * 100) for value in np_map_data]
        list_map_data = [max(0, min(100, v)) for v in list_map_data]
    
        self.map_msg.header.stamp = rclpy.clock.Clock().now().to_msg()
        self.map_msg.data = np.clip((self.mapping.map.flatten() * 100), -128, 127).astype(np.int32).tolist()
        self.map_pub.publish(self.map_msg)
    
        # [6] 10초마다 맵 저장
        current_time = time.time()
        if current_time - self.last_save_time > 10:
            save_map(self, 'update_map.txt')
            self.last_save_time = current_time
    def odom_callback(self, msg):
        """ Odometry 데이터를 받아 현재 방향 (yaw) 업데이트 """
        orientation_q = msg.pose.pose.orientation
        quat = Quaternion(orientation_q.w, orientation_q.x, orientation_q.y, orientation_q.z)
        _, _, self.yaw = quat.to_euler()
        print('odometry info =========', msg.pose.x, msg.pose.y, round(self.yaw, 3))

    def goal_callback(self, msg):
        if msg.header.frame_id == 'map':
            goal_x = msg.pose.position.x
            goal_y = msg.pose.position.y
            print(f"목표 위치 (실제 좌표): x={goal_x:.2f}, y={goal_y:.2f}")
            # 위치변환
            goal_map_x = (goal_x - params_map['MAP_CENTER'][0] + params_map['MAP_SIZE'][0]/2) / params_map['MAP_RESOLUTION']
            goal_map_y = (goal_y - params_map['MAP_CENTER'][1] + params_map['MAP_SIZE'][1]/2) / params_map['MAP_RESOLUTION']
            # 목표 위치 확인
            print(f"맵 좌표계 인덱스: map_x={goal_map_x:.0f}, map_y={goal_map_y:.0f}")


            # 파일 경로 설정
            back_folder = '..'  # 상위 폴더를 지정하려는 경우
            pkg_path = PKG_PATH
            folder_name = 'data'  # 맵을 저장할 폴더 이름
            file_name = 'update_map.txt'  # 파일 이름
            full_path = os.path.join(pkg_path, back_folder, folder_name, file_name)  # 전체 경로 설정

            # 데이터 읽기
            with open(full_path, 'r') as file:
                data = file.read().split()
    
            # 데이터 크기 확인
            grid_size = int(params_map['MAP_SIZE'][0]/params_map['MAP_RESOLUTION'])

            # 1차원 배열을 NxM 크기의 2차원 배열로 변환
            data_array = np.array(data, dtype=int).reshape(grid_size, grid_size)
            self.grid = data_array
            self.rows = len(self.grid)
            self.cols = len((self.grid)[0])
            start = (int(self.map_pose_y), int(self.map_pose_x))
            goal = (int(590.0), int(756.0))
            # goal = (int(132.0), int(188.0))
            path, real_path = self.a_star(start, goal)
            print(real_path)
            # 경로 표시
            if path:
                # 경로를 맵에 빨간색으로 표시
                for p in path:
                    data_array[p[0]][p[1]] = 50  # 경로 표시 (예: 값 50으로 표시)

            # 경로가 제대로 표시되지 않으면 경로를 점으로만 표시
            else:
                print("경로를 찾을 수 없습니다.")
            
            # 만든 path를 publish
            self.publish_global_path(real_path)

            # 시각화 (matplotlib 사용)
            fig, ax = plt.subplots()
            # 먼저 전체 맵을 그립니다
            cax = ax.imshow(data_array, cmap='gray', interpolation='nearest')
            # 경로를 빨간색으로 그립니다
            if path:
                for p in path:
                    ax.plot(p[1], p[0], color='red', marker='o', markersize=2)  # 경로를 빨간색 점으로 표시

            plt.colorbar(cax)  # 색상 막대 추가
            plt.title("A* Pathfinding with Red Path")
            plt.show()





def save_map(node, file_path):
    print("save map start!!!")
    
    # 로직 12 : 맵 저장
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
import rclpy
import numpy as np
from rclpy.node import Node
import os
from geometry_msgs.msg import Pose, PoseStamped, Point
from squaternion import Quaternion
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry,OccupancyGrid,MapMetaData,Path
import math
from math import pi,cos,sin,sqrt
import heapq
from collections import deque
from std_msgs.msg import String
from std_msgs.msg import Bool

import matplotlib
matplotlib.use('Agg')  # GUI 비활성화 (필수)
import matplotlib.pyplot as plt

<<<<<<< HEAD
from .config import params_map, PKG_PATH, MQTT_CONFIG, patrol_path
=======
from .config import params_map, PKG_PATH, MQTT_CONFIG
>>>>>>> 7627d6a354910e00ef5abb1259f7c156ee604e2a
import paho.mqtt.client as mqtt


# a_star 노드는  OccupancyGrid map을 받아 grid map 기반 최단경로 탐색 알고리즘을 통해 로봇이 목적지까지 가는 경로를 생성하는 노드입니다.

# 노드 로직 순서
# 1. publisher, subscriber 만들기
# 2. 파라미터 설정
# 3. 맵 데이터 행렬로 바꾸기
# 4. 위치(x,y)를 map의 grid cell로 변환
# 5. map의 grid cell을 위치(x,y)로 변환
# 6. goal_pose 메시지 수신하여 목표 위치 설정
# 7. grid 기반 최단경로 탐색


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


class a_star(Node):

    def __init__(self):
        super().__init__('a_Star')
        # 로직 1. publisher, subscriber 만들기
        self.map_sub = self.create_subscription(OccupancyGrid, 'map', self.map_callback, 1)
        # self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 1)
        self.subscription = self.create_subscription(LaserScan,'/scan',self.scan_callback,10)
        #self.goal_sub = self.create_subscription(PoseStamped, 'goal_pose', self.goal_callback, 1)
        self.global_path_pub = self.create_publisher(Path, 'a_star_global_path', 10)
        #self.move_order_pub = self.create_publisher(Bool, '/move_order', 1)

        # 이동 타이머 설정
        self.timer = self.create_timer(0.1, self.check_command)

        # MQTT 설정 
        self.mqtt_client = mqtt.Client()
        self.mqtt_broker = MQTT_CONFIG["BROKER"]
        self.mqtt_port = MQTT_CONFIG["PORT"]
        # 목표 경로로 이동 mqtt_topic
        self.mqtt_topic = "robot/destination"
        # 순찰 명령을 받을 mqtt_topic
        self.mqtt_patrol_topic = "robot/patrol"


        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
        self.mqtt_client.loop_start()

        # MQTT 설정 
        self.mqtt_client = mqtt.Client()
        self.mqtt_broker = MQTT_CONFIG["BROKER"]
        self.mqtt_port = MQTT_CONFIG["PORT"]
        self.mqtt_topic = "robot/goal"

        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
        self.mqtt_client.loop_start()


        # 로직 2. 파라미터 설정
        self.map_size_x = int(params_map["MAP_SIZE"][0] / params_map["MAP_RESOLUTION"])
        self.map_size_y = int(params_map["MAP_SIZE"][1] / params_map["MAP_RESOLUTION"])
        self.map_resolution = params_map["MAP_RESOLUTION"]

        self.map_offset_x = params_map["MAP_CENTER"][0] - params_map["MAP_SIZE"][0]/2
        self.map_offset_y = params_map["MAP_CENTER"][1] - params_map["MAP_SIZE"][1]/2

        self.GRIDSIZE=self.map_size_x

        ## 주변 그리드를 탐색할 때 사용할 리스트
        self.dx = [-1,0,0,1,-1,-1,1,1]
        self.dy = [0,1,-1,0,-1,1,-1,1]
        self.dCost = [1,1,1,1,1.414,1.414,1.414,1.414]

        # 현재 위치 및 방향
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.heading = 0.0
        
        self.grid = []
        self.rows = []
        self.cols = []
                
        self.map_pose_x = 0
        self.map_pose_y = 0

<<<<<<< HEAD
        # 순찰 경로 인덱스
        self.patrol_idx = 0
        self.is_patrol_command = False

    # MQTT 연결 시 실행될 콜백 함수
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("✅ MQTT 연결 성공")
            client.subscribe(self.mqtt_topic)
            client.subscribe(self.mqtt_patrol_topic)
        else:
            print(f"❌ MQTT 연결 실패 (코드: {rc})")

    def on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = msg.payload.decode("utf-8")
            print(topic)
            # 요청 받은 경로를 움직이는 경우
            if topic == self.mqtt_topic:
                goal_x, goal_y = map(float, payload.split(","))
                print(f"📌 MQTT 목표 좌표 수신: x={goal_x}, y={goal_y}")

                # MQTT에서 받은 좌표를 맵 좌표계로 변환
                goal_map_x = (goal_x - params_map['MAP_CENTER'][0] + params_map['MAP_SIZE'][0] / 2) / params_map['MAP_RESOLUTION']
                goal_map_y = (goal_y - params_map['MAP_CENTER'][1] + params_map['MAP_SIZE'][1] / 2) / params_map['MAP_RESOLUTION']

                goal_map_x = int(goal_map_x) 
                goal_map_y = int(goal_map_y)

                print(f"📍 변환된 목표 위치 (그리드): x={goal_map_x}, y={goal_map_y}")
                self.path_finding(goal_map_x, goal_map_y)

            # 순찰 명령을 받은 경우
            elif topic == self.mqtt_patrol_topic:
                print(f"📌 순찰 명령 수신: {patrol_path}")
                self.is_patrol_command = True
                goal_x, goal_y = patrol_path[self.patrol_idx]
                # 좌표를 맵 좌표계로 변환
                goal_map_x = int((goal_x - params_map['MAP_CENTER'][0] + params_map['MAP_SIZE'][0] / 2) / params_map['MAP_RESOLUTION'])
                goal_map_y = int((goal_y - params_map['MAP_CENTER'][1] + params_map['MAP_SIZE'][1] / 2) / params_map['MAP_RESOLUTION'])
                print(f"📍 변환된 목표 위치 (그리드): x={goal_map_x}, y={goal_map_y}")
                self.path_finding(goal_map_x, goal_map_y)

        except Exception as e:
            print(f"❌ 목표 좌표 처리 오류: {e}")


    def check_command(self):
        if self.is_patrol_command == False: 
            pass
        else:
            if math.hypot(self.pose_x - patrol_path[self.patrol_idx][0], self.pose_y - patrol_path[self.patrol_idx][1]) < 0.2:
                if self.patrol_idx == len(patrol_path):
                    self.patrol_idx = 0
                    self.is_patrol_command = False
                else:
                    self.patrol_idx += 1
                    goal_x, goal_y = patrol_path[self.patrol_idx]
                    # 좌표를 맵 좌표계로 변환
                    goal_map_x = int((goal_x - params_map['MAP_CENTER'][0] + params_map['MAP_SIZE'][0] / 2) / params_map['MAP_RESOLUTION'])
                    goal_map_y = int((goal_y - params_map['MAP_CENTER'][1] + params_map['MAP_SIZE'][1] / 2) / params_map['MAP_RESOLUTION'])
                    print(f"📍 변환된 목표 위치 (그리드): x={goal_map_x}, y={goal_map_y}")
                    self.path_finding(goal_map_x, goal_map_y)
            else: 
                pass 

=======
        #self.is_order = False

    # def heuristic(self, a, b):
    #     # 맨해튼 거리 (거리 계산 방법을 변경할 수 있음)
    #     return abs(a[0] - b[0]) + abs(a[1] - b[1])
>>>>>>> 7627d6a354910e00ef5abb1259f7c156ee604e2a

    # MQTT 연결 시 실행될 콜백 함수
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("✅ MQTT 연결 성공")
            client.subscribe(self.mqtt_topic)
        else:
            print(f"❌ MQTT 연결 실패 (코드: {rc})")

    def on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8")
            goal_x, goal_y = map(float, payload.split(","))
            print(f"📌 MQTT 목표 좌표 수신: x={goal_x}, y={goal_y}")

            # MQTT에서 받은 좌표를 맵 좌표계로 변환
            goal_map_x = (goal_x - params_map['MAP_CENTER'][0] + params_map['MAP_SIZE'][0] / 2) / params_map['MAP_RESOLUTION']
            goal_map_y = (goal_y - params_map['MAP_CENTER'][1] + params_map['MAP_SIZE'][1] / 2) / params_map['MAP_RESOLUTION']

            goal_map_x = int(goal_map_x) 
            goal_map_y = int(goal_map_y)

            print(f"📍 변환된 목표 위치 (그리드): x={goal_map_x}, y={goal_map_y}")

            # 맵 데이터 로드
            back_folder = '..'  # 상위 폴더 지정
            pkg_path = PKG_PATH
            folder_name = 'data'
            file_name = 'update_map.txt'
            full_path = os.path.join(pkg_path, back_folder, folder_name, file_name)

            # 데이터 읽기
            with open(full_path, 'r') as file:
                data = file.read().split()

            # 그리드 크기 계산
            grid_size = int(params_map['MAP_SIZE'][0] / params_map['MAP_RESOLUTION'])
            print(f"그리드 사이즈: {grid_size} x {grid_size}")

            # 데이터 크기 불일치 확인
            if len(data) != grid_size * grid_size:
                print("⚠ 데이터 크기가 맞지 않습니다! 파일 데이터 개수와 그리드 크기가 일치하는지 확인하세요.")
                return

            # 1차원 배열을 NxM 크기의 2차원 배열로 변환
            # data_array = np.array(data, dtype=int).flatten().reshape(grid_size, grid_size)
            data_array = np.array(data, dtype=int).reshape(grid_size, grid_size)

            # 맵 좌표 인덱스 범위 초과 방지
            if not (0 <= goal_map_x < data_array.shape[0] and 0 <= goal_map_y < data_array.shape[1]):
                print(f"⚠ 오류: goal_map_x={goal_map_x}, goal_map_y={goal_map_y}가 data_array 범위를 초과합니다.")
                return

            self.grid = data_array
            self.rows, self.cols = data_array.shape

            # A* 실행
            start = (int(self.map_pose_y), int(self.map_pose_x))
            print(start)
            goal = (goal_map_y, goal_map_x)

            path, real_path = self.a_star(start, goal)
            print(real_path)
            print("끝!!!")
            if path:
                print(f"✅ 경로 탐색 성공! 경로 길이: {len(path)}")
                #move_order_msg = Bool()
                #move_order_msg.data = True
                #self.move_order_pub.publish(move_order_msg)

                self.publish_global_path(real_path)
                

            else:
                print("⚠️ 경로를 찾을 수 없음.")

        except Exception as e:
            print(f"❌ 목표 좌표 처리 오류: {e}")

    def heuristic(self, a, b):
        #print("heuristic!!")
        base_heuristic = abs(a[0] - b[0]) + abs(a[1] - b[1])  # 맨해튼 거리
        safety_penalty = 0
    
        # 벽 근처에 있을 경우 패널티 추가 (4칸)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = a[0] + dx, a[1] + dy
                #print(self.rows,'rows')
                if 0 <= nx < self.rows and 0 <= ny < self.cols:
                    if self.grid[nx, ny] >= 40:  # 벽이면
                        safety_penalty += 5  # 패널티 크게 증가
        
        #print("통과")
        return base_heuristic + safety_penalty


    def neighbors(self, node):
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
        dCost = [1, 1, 1, 1, 1.414, 1.414, 1.414, 1.414]  # 대각선 이동 비용

        neighbors = []
        for i, direction in enumerate(directions):
            nx, ny = node[0] + direction[0], node[1] + direction[1]
            
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                cost = dCost[i]
                
                # 벽 근처 가중치 추가 (4칸 안전 마진 적용)
                min_distance = 2  
                for dx in range(-min_distance, min_distance + 1):
                    for dy in range(-min_distance, min_distance + 1):
                        if 0 <= nx + dx < self.rows and 0 <= ny + dy < self.cols:
                            if self.grid[nx + dx, ny + dy] >= 40:  # 벽 근처
                                cost += 5  # 패널티 더 증가
                
                if self.grid[nx, ny] < 40:  # 벽이 아닌 경우에만 추가
                    neighbors.append(((nx, ny), cost))
        
        return neighbors

    
    def a_star(self, start, goal):
        print("a star 시작")
        # 좌표 정수 변환
        start = (int(round(start[0])), int(round(start[1])))
        goal = (int(round(goal[0])), int(round(goal[1])))
        print(start)
        print(goal)

        open_list = []
        closed_list = set()

        heapq.heappush(open_list, (0 + self.heuristic(start, goal), 0, self.heuristic(start, goal), start))
        print(open_list,'open_list')
        came_from = {}
        g_score = {start: 0}

        while open_list:
            current_f, current_g, current_h, current_node = heapq.heappop(open_list)
            #print("while 문")
            #print(current_node, 'current_node')
            #print(goal,'goal')
            if current_node == goal:
                print("✅ 목표 도착!")
                path = []
                #print(came_from,'came_from')
                while current_node in came_from:
                    path.append(current_node)
                    current_node = came_from[current_node]
                path.append(start)
                real_path = grid_to_real(path[::-1], params_map)
                return path[::-1], real_path

            closed_list.add(current_node)

            for neighbor, cost in self.neighbors(current_node):
                #print(f"  ↪️ 이웃 노드: {neighbor}, 방문 여부: {neighbor in closed_list}")
                neighbor = (int(round(neighbor[0])), int(round(neighbor[1])))  # 정수 변환 추가
                
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
        print("경로를 Path 메시지로 변환 후 Publish")
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


    def grid_cell_to_pose(self, grid_cell):
        ## 로직 5. map의 grid cell을 위치(x,y)로 변환
        x = grid_cell[0] * self.map_resolution + self.map_offset_x
        y = grid_cell[1] * self.map_resolution + self.map_offset_y
        return [x, y]



    def path_finding(self, goal_map_x, goal_map_y):
        # 맵 데이터 로드
        back_folder = '..'  # 상위 폴더 지정
        pkg_path = PKG_PATH
        folder_name = 'data'
        file_name = 'update_map.txt'
        full_path = os.path.join(pkg_path, back_folder, folder_name, file_name)

        # 데이터 읽기
        with open(full_path, 'r') as file:
            data = file.read().split()

        # 그리드 크기 계산
        grid_size = int(params_map['MAP_SIZE'][0] / params_map['MAP_RESOLUTION'])
        print(f"그리드 사이즈: {grid_size} x {grid_size}")

        # 데이터 크기 불일치 확인
        if len(data) != grid_size * grid_size:
            print("⚠ 데이터 크기가 맞지 않습니다! 파일 데이터 개수와 그리드 크기가 일치하는지 확인하세요.")
            return

        # 1차원 배열을 NxM 크기의 2차원 배열로 변환
        data_array = np.array(data, dtype=int).reshape(grid_size, grid_size)

        # 맵 좌표 인덱스 범위 초과 방지
        if not (0 <= goal_map_x < data_array.shape[0] and 0 <= goal_map_y < data_array.shape[1]):
            print(f"⚠ 오류: goal_map_x={goal_map_x}, goal_map_y={goal_map_y}가 data_array 범위를 초과합니다.")
            return

        self.grid = data_array
        self.rows, self.cols = data_array.shape

        # A* 실행
        start = (int(self.map_pose_y), int(self.map_pose_x))
        print(start)
        goal = (goal_map_y, goal_map_x)

        path, real_path = self.a_star(start, goal)
        print(real_path)
        print("끝!!!")
        if path:
            print(f"✅ 경로 탐색 성공! 경로 길이: {len(path)}")
            self.publish_global_path(real_path)
            for p in path:
                data_array[p[0]][p[1]] = 50  # 경로 표시
        else:
            print("⚠️ 경로를 찾을 수 없음.")

        # 만든 path를 publish
        self.publish_global_path(real_path)

        # 시각화
        fig, ax = plt.subplots()
        cax = ax.imshow(data_array, cmap='gray', interpolation='nearest')
        if path:
            for p in path:
                ax.plot(p[1], p[0], color='red', marker='o', markersize=2)
        plt.colorbar(cax)
        plt.title("A* Pathfinding with Red Path")
        # 저장 경로 구성
        back_folder = '..'
        pkg_path = PKG_PATH
        folder_name = 'data'
        save_filename = 'a_star_result.png'

        save_dir = os.path.join(pkg_path, back_folder, folder_name)
        os.makedirs(save_dir, exist_ok=True)  # 폴더 없으면 생성
        save_path = os.path.join(save_dir, save_filename)

        # 이미지 저장
        plt.savefig(save_path)
        plt.close()
        print(f"📷 시각화 이미지 저장 완료: {save_path}")

        # 이미지 열기
        os.startfile(save_path)

    def odom_callback(self, msg):
        self.is_odom = True
        self.odom_msg = msg

    def map_callback(self, msg):
        self.is_map = True
        self.map_msg = msg

    def scan_callback(self, msg):
        # print("scan_callback start!!!")
    
        # [1] 현재 위치 (pose_x, pose_y, heading) 가져오기
        self.pose_x = msg.range_min  # 실제 x 좌표 (meters)
        self.pose_y = msg.scan_time  # 실제 y 좌표 (meters)
        self.heading = msg.time_increment  # 로봇의 방향 (radians)

        self.map_pose_x = (self.pose_x - params_map['MAP_CENTER'][0] + params_map['MAP_SIZE'][0]/2) / params_map['MAP_RESOLUTION']
        self.map_pose_y = (self.pose_y - params_map['MAP_CENTER'][1] + params_map['MAP_SIZE'][1]/2) / params_map['MAP_RESOLUTION']
        #print(f'현 위치: {self.map_pose_x, self.map_pose_y} ')


def main(args=None):
    rclpy.init(args=args)
    global_planner = a_star()
    rclpy.spin(global_planner)
    global_planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

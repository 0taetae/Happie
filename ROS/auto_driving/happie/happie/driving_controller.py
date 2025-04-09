import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import numpy as np
import time, json

from .config import params_map, PKG_PATH, MQTT_CONFIG
import paho.mqtt.client as mqtt
from std_msgs.msg import Int32, String

class Controller(Node):
    def __init__(self):
        super().__init__('controller')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 1)
        self.a_star_global_path_sub = self.create_subscription(Path, '/a_star_global_path', self.global_path_callback, 1)
        self.order_id_sub = self.create_subscription(Int32, '/order_id', self.order_id_callback, 1)
        self.priority_work_sub = self.create_subscription(String, '/priority_work', self.priority_work_callback, 1)

        self.equipment_detected_sub = self.create_subscription(Int32, '/equipment_detected', self.equipment_callback, 1)
        #self.move_order_sub = self.create_subscription(Bool, '/move_order', self.move_order_callback, 1)
        #self.move_order_pub = self.create_publisher(Bool, '/move_order', 1)
        #self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_request_pub = self.create_publisher(Point, '/request_new_path', 1) # 장애물 감지 시 새 경로 요청
        self.fall_sub = self.create_subscription(Bool,'/fall_detected',self.fall_callback, 1) # 낙상 감지 
        self.cmd_msg = Twist()

        # 현재 위치 및 방향
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.heading = 0.0  # LaserScan에서 계산

        # 배터리 잔량
        self.is_charging = False
        self.cum_pose = []
        self.prior_pose = 0.0
        self.present_pose = 0.0
        self.battery = 1000.0

        # 이동 타이머 설정
        self.timer = self.create_timer(0.1, self.move_to_destination)

        self.is_to_move = False
        self.fall_detected = False
        #self.is_order = False
        # a_star를 통해 생성한 global_path
        self.global_path = [(-51.0,-51.0)]
        # self.global_path = [(-51.412, -52.948), (-51.412, -52.977999999999994), (-51.412, -53.007999999999996), (-51.412, -53.038), (-51.442, -53.068), (-51.472, -53.098), (-51.502, -53.128), (-51.532000000000004, -53.158), (-51.562000000000005, -53.187999999999995), (-51.592, -53.217999999999996), (-51.622, -53.248), (-51.652, -53.278), (-51.682, -53.308), (-51.712, -53.338), (-51.712, -53.367999999999995), (-51.712, -53.397999999999996), (-51.712, -53.428), (-51.712, -53.458), (-51.712, -53.488), (-51.712, -53.518), (-51.712, -53.547999999999995), (-51.712, -53.577999999999996), (-51.712, -53.608), (-51.712, -53.638), (-51.712, -53.668), (-51.712, -53.698), (-51.712, -53.727999999999994), (-51.712, -53.757999999999996), (-51.742000000000004, -53.788), (-51.742000000000004, -53.818), (-51.742000000000004, -53.848), (-51.742000000000004, -53.878), (-51.742000000000004, -53.908), (-51.742000000000004, -53.937999999999995), (-51.742000000000004, -53.967999999999996), (-51.742000000000004, -53.998), (-51.742000000000004, -54.028), (-51.772000000000006, -54.058), (-51.802, -54.087999999999994), (-51.832, -54.117999999999995), (-51.862, -54.147999999999996), (-51.892, -54.178), (-51.922000000000004, -54.208), (-51.952000000000005, -54.238), (-51.982, -54.268), (-52.012, -54.297999999999995), (-52.042, -54.327999999999996), (-52.072, -54.358), (-52.102000000000004, -54.388), (-52.132000000000005, -54.418), (-52.162000000000006, -54.448), (-52.192, -54.477999999999994), (-52.222, -54.507999999999996), (-52.252, -54.538), (-52.282000000000004, -54.568), (-52.312000000000005, -54.598), (-52.342, -54.628), (-52.372, -54.628), (-52.402, -54.628), (-52.432, -54.628), (-52.462, -54.628), (-52.492000000000004, -54.658), (-52.522000000000006, -54.658), (-52.552, -54.658), (-52.582, -54.658), (-52.612, -54.658), (-52.642, -54.658), (-52.672000000000004, -54.658), (-52.702000000000005, -54.658), (-52.732, -54.658), (-52.762, -54.658), (-52.792, -54.658), (-52.822, -54.658), (-52.852000000000004, -54.658), (-52.882000000000005, -54.658), (-52.912, -54.658), (-52.942, -54.658), (-52.972, -54.658), (-53.002, -54.628), (-53.032000000000004, -54.598), (-53.032000000000004, -54.568), (-53.032000000000004, -54.538), (-53.032000000000004, -54.507999999999996), (-53.032000000000004, -54.477999999999994), (-53.032000000000004, -54.448), (-53.032000000000004, -54.418), (-53.032000000000004, -54.388), (-53.032000000000004, -54.358), (-53.062000000000005, -54.327999999999996), (-53.092, -54.297999999999995), (-53.122, -54.268), (-53.152, -54.238), (-53.182, -54.208), (-53.212, -54.178), (-53.242000000000004, -54.147999999999996), (-53.272000000000006, -54.117999999999995), (-53.302, -54.087999999999994), (-53.302, -54.058), (-53.302, -54.028), (-53.302, -53.998), (-53.302, -53.967999999999996), (-53.302, -53.937999999999995), (-53.302, -53.908), (-53.302, -53.878), (-53.302, -53.848), (-53.302, -53.818), (-53.302, -53.788), (-53.332, -53.757999999999996), (-53.332, -53.727999999999994), (-53.332, -53.698), (-53.332, -53.668), (-53.332, -53.638), (-53.332, -53.608), (-53.332, -53.577999999999996), (-53.332, -53.547999999999995), (-53.332, -53.518), (-53.332, -53.488), (-53.332, -53.458), (-53.332, -53.428), (-53.332, -53.397999999999996), (-53.332, -53.367999999999995), (-53.332, -53.338), (-53.332, -53.308), (-53.332, -53.278), (-53.332, -53.248), (-53.332, -53.217999999999996), (-53.332, -53.187999999999995), (-53.332, -53.158), (-53.332, -53.128), (-53.332, -53.098), (-53.332, -53.068), (-53.332, -53.038), (-53.332, -53.007999999999996), (-53.332, -52.977999999999994), (-53.332, -52.948), (-53.302, -52.918), (-53.302, -52.888), (-53.272000000000006, -52.858), (-53.242000000000004, -52.827999999999996), (-53.212, -52.797999999999995), (-53.212, -52.768), (-53.212, -52.738), (-53.212, -52.708), (-53.212, -52.678), (-53.212, -52.647999999999996), (-53.212, -52.617999999999995), (-53.212, -52.588), (-53.212, -52.558), (-53.212, -52.528), (-53.212, -52.498), (-53.212, -52.467999999999996), (-53.212, -52.437999999999995), (-53.212, -52.408), (-53.212, -52.378), (-53.212, -52.348), (-53.212, -52.318), (-53.212, -52.288), (-53.212, -52.257999999999996), (-53.212, -52.227999999999994), (-53.212, -52.198), (-53.212, -52.168), (-53.212, -52.138), (-53.212, -52.108), (-53.212, -52.077999999999996), (-53.212, -52.047999999999995), (-53.212, -52.018), (-53.212, -51.988), (-53.212, -51.958), (-53.212, -51.928), (-53.212, -51.897999999999996), (-53.212, -51.867999999999995), (-53.212, -51.838), (-53.212, -51.808), (-53.212, -51.778), (-53.212, -51.748), (-53.212, -51.717999999999996), (-53.212, -51.687999999999995), (-53.212, -51.658), (-53.212, -51.628), (-53.212, -51.598), (-53.212, -51.568), (-53.182, -51.538), (-53.152, -51.507999999999996), (-53.122, -51.477999999999994), (-53.092, -51.448), (-53.062000000000005, -51.418), (-53.032000000000004, -51.388), (-53.002, -51.358), (-52.972, -51.327999999999996), (-52.942, -51.297999999999995), (-52.912, -51.268), (-52.882000000000005, -51.238), (-52.852000000000004, -51.208), (-52.822, -51.178), (-52.792, -51.147999999999996), (-52.762, -51.117999999999995), (-52.732, -51.088), (-52.702000000000005, -51.058), (-52.672000000000004, -51.028), (-52.642, -50.998), (-52.612, -50.967999999999996), (-52.582, -50.937999999999995), (-52.552, -50.908), (-52.522000000000006, -50.878), (-52.492000000000004, -50.848), (-52.462, -50.818), (-52.432, -50.788), (-52.402, -50.757999999999996), (-52.372, -50.727999999999994), (-52.342, -50.698), (-52.312000000000005, -50.668), (-52.282000000000004, -50.638), (-52.252, -50.608), (-52.222, -50.577999999999996), (-52.192, -50.547999999999995), (-52.162000000000006, -50.518), (-52.132000000000005, -50.488), (-52.102000000000004, -50.458), (-52.072, -50.428), (-52.042, -50.397999999999996), (-52.012, -50.367999999999995), (-51.982, -50.338), (-51.952000000000005, -50.308), (-51.922000000000004, -50.278), (-51.892, -50.248), (-51.862, -50.217999999999996), (-51.832, -50.187999999999995), (-51.802, -50.158), (-51.772000000000006, -50.128), (-51.742000000000004, -50.098), (-51.712, -50.068), (-51.682, -50.038), (-51.652, -50.038), (-51.622, -50.038), (-51.592, -50.038), (-51.562000000000005, -50.038), (-51.532000000000004, -50.038), (-51.502, -50.038), (-51.472, -50.038), (-51.442, -50.038), (-51.412, -50.038), (-51.382000000000005, -50.038), (-51.352000000000004, -50.038), (-51.322, -50.038), (-51.292, -50.038), (-51.262, -50.038), (-51.232, -50.038), (-51.202000000000005, -50.038), (-51.172000000000004, -50.038), (-51.142, -50.038), (-51.112, -50.038), (-51.082, -50.038), (-51.052, -50.038), (-51.022000000000006, -50.038), (-50.992000000000004, -50.038), (-50.962, -50.038), (-50.932, -50.038), (-50.902, -50.007999999999996), (-50.872, -50.007999999999996), (-50.842000000000006, -50.007999999999996), (-50.812000000000005, -50.007999999999996), (-50.782000000000004, -50.007999999999996), (-50.752, -50.007999999999996), (-50.722, -50.007999999999996), (-50.692, -50.007999999999996), (-50.662, -50.007999999999996), (-50.632000000000005, -50.007999999999996), (-50.602000000000004, -50.007999999999996), (-50.572, -50.007999999999996), (-50.542, -50.007999999999996), (-50.512, -50.007999999999996), (-50.482, -50.007999999999996), (-50.452000000000005, -50.007999999999996), (-50.422000000000004, -50.007999999999996), (-50.392, -50.007999999999996), (-50.362, -50.007999999999996), (-50.332, -50.007999999999996), (-50.302, -50.007999999999996), (-50.272000000000006, -50.007999999999996), (-50.242000000000004, -50.007999999999996), (-50.212, -50.007999999999996), (-50.182, -50.007999999999996), (-50.152, -50.007999999999996), (-50.122, -50.007999999999996), (-50.092000000000006, -50.007999999999996), (-50.062000000000005, -50.007999999999996), (-50.032000000000004, -50.007999999999996), (-50.002, -50.007999999999996)]
        # self.global_path = self.global_path[::6]
        self.current_goal_idx = 0

        # 우선 명령 변수
        self.is_priority_work = False
        self.type_priority_work = ''

        # 목표 지점 설정
        self.goal = Point()
        self.set_new_goal()
        self.object_detected = False
        self.object_detected_cnt = 1000
        self.path_requested = False
        self.object_angle = 0.0
        self.order_id = None 

        # MQTT 설정 
        self.mqtt_client = mqtt.Client()
        self.mqtt_broker = MQTT_CONFIG["BROKER"]
        self.mqtt_port = MQTT_CONFIG["PORT"]
        self.mqtt_topic_log = "robot/log"
        self.mqtt_topic_fall_check = "robot/fall_check" # 낙상 확인 후 재이동 

        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
        self.mqtt_client.loop_start()
    
    # MQTT 연결 시 실행될 콜백 함수
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("✅ MQTT 연결 성공(driving_controller)")
            client.subscribe(self.mqtt_topic_fall_check)
        else:
            print(f"❌ MQTT 연결 실패 (코드: {rc})")
    
    def on_message(self, client, userdata, msg):
        payload = msg.payload.decode("utf-8")
        data = json.loads(payload)
        print(f"📨 MQTT 메시지 수신: {payload}")
        if data["status"] == "check":
            if self.fall_detected:
                print("✅ 낙상 해제 신호 수신 → 이동 재개")
                self.fall_detected = False
                self.is_to_move = True
    
    def scan_callback(self, msg):
        self.cum_pose.append(msg)
        # 매 초당, 대기전력 0.01 사용
        self.battery -= 0.01
        self.battery = max(self.battery, 0.0)
        # 위치 초기값 설정
        if self.prior_pose == 0.0:
            self.prior_pose = (msg.range_min, msg.scan_time)
            self.present_pose = (msg.range_min, msg.scan_time)
        # 위치 업데이트
        else:
            self.prior_pose, self.present_pose = self.present_pose, (msg.range_min, msg.scan_time)
        # 이동거리 측정        
        moved_dist = math.hypot((self.prior_pose[0] - self.present_pose[0]), (self.prior_pose[1] - self.present_pose[1]))
        self.battery -= moved_dist/3

        # LaserScan 데이터를 받아 현재 위치와 heading 업데이트 
        self.pose_x = msg.range_min
        self.pose_y = msg.scan_time 
        self.ranges = np.array(msg.ranges)
        # print([round(val, 2) for val in msg.ranges])

        self.heading = (msg.time_increment + 360) % 360
        # 일정 시간 지나기 전, 다시 장애물 감지 하지않음.
        if self.object_detected_cnt >= 0: return

        left = [val for val in self.ranges[:20] if val < 2.0]
        right = [val for val in self.ranges[339:359] if val < 2.0]
        front = [val for val in  self.ranges[:10] if val < 2.0] + [val for val in self.ranges[349:359] if val < 2.0]
        left = sum(left) / len(left) if len(left) else 100
        right = sum(right) / len(right) if len(right) else 100
        front = sum(front) / len(front) if len(front) else 100
        pivot = min(front, right, left)
        # print(msg.ranges)

        if pivot < 0.3: 
            print(f'장애물 감지됨, 재 감지까지 남은시간: {self.object_detected_cnt}')
            if self.object_detected == False:
                self.object_detected = True
                # if front == pivot: print('정면 장애물 감지')
                # elif right == pivot: print('우측면 장애물 감지')
                # elif left == pivot: print('좌측면 장애물 감지')

                self.turtlebot_stop() 
                self.request_new_path()
                self.path_requested = True  # 한 번만 요청하도록 설정
                self.object_detected_cnt = 1000
        else:
            self.object_detected = False


    def order_id_callback(self, msg):
        self.order_id = msg.data
        print(f"명령 ID 수신: {self.order_id}")


    def priority_work_callback(self, msg):
        self.type_priority_work = msg.data
        print(f"우선순위 명령 수신: {self.type_priority_work}")        

    def equipment_callback(self, msg):
        if msg.data:  # 기자재 감지됨
            self.is_priority_work = True
            print("기자재 감지!")


    def fall_callback(self,msg):
        self.fall_detected = msg.data
        print(self.fall_detected,'callback')
        if self.fall_detected:
            print("🛑 낙상 감지됨 → 이동 정지")
            self.is_to_move = False
            self.turtlebot_stop()
        else:
            print("✅ 낙상 해제")


    def global_path_callback(self, msg):
        path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.global_path = path
        self.goal.x = path[0][0]
        self.goal.y = path[0][1]
        # print(self.global_path)
        print("경로 받기 성공")
        self.current_goal_idx = 0
        self.is_to_move = True
        self.mqtt_client.publish(self.mqtt_topic_log, "moving")
        self.path_requested = False


    # 장애물 감지 시 새로운 경로를 요청하고 목적지 좌표를 전달
    def request_new_path(self, type='', new_goal = (-1, -1)):
        # 메시지 생성 (목적지 좌표 포함)
        path_request_msg = Point()
        # 충전소 보내기
        if type == 'charge':
            path_request_msg.x = -42.44
            path_request_msg.y = -45.60
            path_request_msg.z = self.object_angle
        
        elif type == 'new_goal':
            pass
        else:
            print(f"📢 새로운 경로 요청! 목적지: ({self.global_path[-1][0]}, {self.global_path[-1][1]})")
            path_request_msg.x = self.global_path[-1][0]
            path_request_msg.y = self.global_path[-1][1]
            path_request_msg.z = self.object_angle

        # A* 노드에 경로 요청
        self.path_request_pub.publish(path_request_msg)

        # 이동 중지
        self.is_to_move = False 

    def set_new_goal(self):
        self.turtlebot_stop()
        print(self.current_goal_idx, ' 인덱스')

        # global_path가 비어 있지 않을 때만 진행
        if self.current_goal_idx < len(self.global_path):
            # 경로에서 앞으로 10개의 포인트 중 현재 위치에서 가장 가까운 지점 선택
            search_window = self.global_path[self.current_goal_idx:self.current_goal_idx + 10]
            min_dist = float('inf')
            closest_idx = self.current_goal_idx  # 초기화
            for idx, (x, y) in enumerate(search_window):
                dist = math.hypot(x - self.pose_x, y - self.pose_y)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = self.current_goal_idx + idx
            # 가장 가까운 지점으로 목표 설정
            self.goal.x, self.goal.y = self.global_path[closest_idx]
            self.current_goal_idx = closest_idx
            print(f"새 목표 지점 설정: {self.goal.x:.2f}, {self.goal.y:.2f} (인덱스: {self.current_goal_idx})")
        else:
            self.turtlebot_stop()
            self.get_logger().info("finish =========")
            self.is_to_move = False
            self.current_goal_idx = 0
            # rclpy.shutdown()

            # 기자재 방에 도착했을 경우 
            if self.pose_x <= -55.20 and self.pose_y <= -51.76:
                payload = {
                    "id": self.order_id if self.order_id is not None else -1,
                    "status": "arrive"
                }
            else:
                payload = {
                    "id": self.order_id if self.order_id is not None else -1,
                    "status": "finish"
                }
            self.mqtt_client.publish(self.mqtt_topic, json.dumps(payload))
            self.order_id = None

    def move_to_destination(self):
        # print(f'배터리 잔량 {round(self.battery, 2)}%')
        self.object_detected_cnt -= 1
        # 우선순위 작업이 없는 경우
        if self.is_priority_work == False:
            # if self.path_requested == False:
            if self.path_requested == 1:
                if self.battery < 10.0 and self.is_charging == False:
                    self.turtlebot_stop() 
                    self.request_new_path('charge')
                    self.path_requested = True  # 한 번만 요청하도록 설정
                    return
                if self.is_charging:
                    if math.hypot(self.pose_x - -42.44, self.pose_y - 45.6) < 5:
                        # 배터리 충전
                        self.battery += 1.0
                        self.battery = min(self.battery, 100.0)
                        #  배터리가 충전 중이면서, 배터리 잔량이 50% 미만인 경우, 다른 명령 수행 불가능
                        if self.battery < 50.0:
                            return
                    
            vel_msg = Twist()
            if self.is_to_move == False: 
                vel_msg.angular.z = 0.0
                vel_msg.linear.x = 0.0
            # 🚨 장애물이 감지되면 이동을 멈추고 새로운 경로 요청
            else:
                if self.object_detected and (self.path_requested):
                    return 
                else:
                    # 현재 목표까지의 거리 계산
                    distance = math.sqrt((self.goal.x - self.pose_x) ** 2 + (self.goal.y - self.pose_y) ** 2)
                    # 목표 지점 도착 여부 확인
                    if distance < 0.1:
                        # self.get_logger().info(f"목표 지점 {self.current_goal_idx} 도착. 잠시 정지합니다.")
                        print(f"목표 지점 {self.current_goal_idx} 도착. 잠시 정지합니다.")
                        # 목표 지점 도착 후 1초 정지
                        self.turtlebot_stop()
                        self.current_goal_idx += 1
                        self.set_new_goal()

                        return

                    # 목표 heading 계산
                    target_heading = math.degrees(math.atan2(-(self.goal.x - self.pose_x), self.goal.y - self.pose_y))
                    target_heading = (target_heading + 360) % 360  # 0~360도로 변환
                    # print(f'target_heading: {target_heading}')
                    # print(f'현 위치: {round(self.pose_x, 2)} {round(self.pose_y, 2)}')
                    # print(f'현 위치: {round(self.goal.x, 2)} {round(self.goal.y, 2)}')

                    # angle_diff = (target_heading - self.heading + 540) % 360 - 180
                    # 현재 heading과 목표 heading 비교 (최단 회전 경로 고려)
                    if 0.0 <= self.heading <= 180.0:
                        if self.heading <= target_heading <= self.heading + 180.0: weight = -1 
                        else: weight = 1
                    else:
                        if self.heading - 180.0 <= target_heading <= self.heading: weight = 1
                        else: weight = -1

                    angle_diff = abs(self.heading - target_heading)
                    if abs(target_heading -self.heading) < abs(self.heading - target_heading): 
                        angle_diff = abs(target_heading -self.heading)

                    # print('목표 heading:', target_heading, '현재 heading:', self.heading)
                    # 🔹 heading이 목표와 10도 이상 차이나면 회전
                    if angle_diff > 10:

                        # # 회전 속도를 angle_diff에 비례하도록 조정 (단, 최대 속도 제한)
                        if angle_diff <= 20.0:
                            vel_msg.angular.z = 0.08*weight
                        elif angle_diff <= 30.0:
                            vel_msg.angular.z = 0.08*weight
                        elif angle_diff <= 60.0:
                            vel_msg.angular.z = max(0.1, 0.01*(angle_diff/4))*weight
                        elif angle_diff <= 90.0:
                            vel_msg.angular.z = max(0.15, 0.01*(angle_diff/5))*weight
                        elif angle_diff <= 120.0:
                            vel_msg.angular.z = max(0.18, 0.01*(angle_diff/6))*weight
                        else:
                            vel_msg.angular.z = 0.25*weight
                        vel_msg.linear.x = 0.0  # 회전 중 직진 금지
                
                        # =======================================================================

                    else:
                        vel_msg.linear.x = 0.2
                        vel_msg.angular.z = 0.0  # 직진 시 회전 없음

            self.pub.publish(vel_msg)
        # 우선순위 작업이 존재하는 경우
        else:
            pass

    def turtlebot_stop(self):
        # self.get_logger().info("Turtlebot stopping")
        print("=================정지================")
        self.cmd_msg.linear.x = 0.0
        self.cmd_msg.angular.z = 0.0
        self.pub.publish(self.cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
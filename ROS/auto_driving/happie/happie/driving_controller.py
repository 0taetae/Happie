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
        self.battery = 10000.0
        # self.battery = 30.0
        self.is_going_charging_station = False

        # 이동 타이머 설정
        self.timer = self.create_timer(0.01, self.move_to_destination)

        self.is_to_move = False
        self.fall_detected = False

        # a_star를 통해 생성한 global_path
        self.global_path = [(-51.0,-51.0)]
        self.current_goal_idx = 0

        # 우선 명령 변수
        self.correct_Gaussian_error = {'status': False, 'target_heading': 0}
        self.is_priority_work = False
        self.type_priority_work = ''

        # 목표 지점 설정
        self.goal = Point()
        self.set_new_goal()
        self.object_detected = False
        self.object_detected_cnt = 100
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
        # print(self.pose_x, self.pose_y, self.heading)

        # 충전 상태관리
        if math.hypot(-42.44 - self.pose_x, -45.6 - self.pose_y) < 0.1:
            self.is_charging = True if self.battery < 95.0 else False
            if self.is_going_charging_station == True: 
                self.is_going_charging_station = False
        else:
            self.is_charging = False

        self.cum_pose.append(msg)
        # 매 초당, 대기전력 0.01 사용
        self.battery -= 0.01
        # self.battery -= 0.01
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

        # print(self.pose_x, self.pose_y, self.heading)

        # 일정 시간 지나기 전, 다시 장애물 감지 하지않음.
        if self.object_detected_cnt >= 0: return

        left = [val for val in self.ranges[:20] if val < 2.0]
        right = [val for val in self.ranges[339:359] if val < 2.0]
        front = [val for val in  self.ranges[:10] if val < 2.0] + [val for val in self.ranges[349:359] if val < 2.0]
        left = sum(left) / len(left) if len(left) else 100
        right = sum(right) / len(right) if len(right) else 100
        front = sum(front) / len(front) if len(front) else 100
        pivot = min(front, right, left)
        # print(pivot)
        # print(msg.ranges)

        if 0.05 < pivot < 0.2: 
            print(f'장애물 감지됨, 재 감지까지 남은시간: {self.object_detected_cnt}')

            if self.object_detected == False:
                self.is_to_move = False
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
        # self.order_id = msg.data
        # print(f"명령 ID 수신: {self.order_id}")
        return

    def priority_work_callback(self, msg):
        self.type_priority_work = msg.data
        print(f"우선순위 명령 수신: {self.type_priority_work}")        

    def equipment_callback(self, msg):
        if msg.data:  # 기자재 감지됨
            self.is_priority_work = True
            print(f"기자재 감지! {[None, 'WheelChair', 'Intravenous'][msg.data]}")


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
        print('global path callback ==========')
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
            print('충전명령 받음')
            path_request_msg.x = -42.44
            path_request_msg.y = -45.60
            path_request_msg.z = self.object_angle
            self.is_going_charging_station = True
        
        elif type == 'new_goal':
            pass

        else:
            path_request_msg.x = self.global_path[-1][0]
            path_request_msg.y = self.global_path[-1][1]
            path_request_msg.z = self.object_angle
            print('장애물 감지로 인한, 새로운 경로 요청')

        # A* 노드에 경로 요청
        self.path_request_pub.publish(path_request_msg)

        # 이동 중지
        self.is_to_move = False 

    def set_new_goal(self):
        self.turtlebot_stop()
        # print(self.current_goal_idx, ' 인덱스')

        # global_path가 비어 있지 않을 때만 진행
        if self.current_goal_idx < len(self.global_path):
            # 경로에서 앞으로 10개의 포인트 중 현재 위치에서 가장 가까운 지점 선택
            search_window = self.global_path[self.current_goal_idx:self.current_goal_idx + 15]
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
            # print(f"새 목표 지점 설정: {self.goal.x:.2f}, {self.goal.y:.2f} (인덱스: {self.current_goal_idx})")
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
            self.mqtt_client.publish(self.mqtt_topic_log, json.dumps(payload))
            self.order_id = None

    def move_to_destination(self):
        # print(self.is_to_move)
        # print("self.path_requested, self.is_charging, self.is_going_charging_station")
        # print(self.path_requested, self.is_charging, self.is_going_charging_station)
        # print(f'배터리 잔량 {round(self.battery, 2)}%')
        self.object_detected_cnt -= 0.2

        # heading 오차 보정 ==============================
        if self.correct_Gaussian_error['status'] == True:
            vel_msg = Twist()
            # 각도 차이 구하기 STEP 1
            Ang_1, Ang_2 = self.heading, self.correct_Gaussian_error['target_heading']
            if Ang_1 < Ang_2: Ang_1, Ang_2 = Ang_2, Ang_1
            # 각도 차이 구하기 STEP 2
            angle_diff = abs(Ang_1 - Ang_2)
            if abs(Ang_1 - (Ang_2 + 360.0)) < angle_diff:
                angle_diff = abs(Ang_1 - (Ang_2 + 360.0))
            # print('각도 보정중 =====')
            # print(self.correct_Gaussian_error['target_heading'], self.heading)
            if angle_diff < 1:
                # print('각도 차이 1도 미만')
                vel_msg.linear.x = 0.2
                vel_msg.angular.z = 0.0  # 직진 시 회전 없음
                # heading 오차 보정 상태 해제
                self.correct_Gaussian_error['status'] = False

            else:
                if angle_diff < 15.0:
                    vel_msg.angular.z = 0.1*self.correct_Gaussian_error['weight']
                    vel_msg.linear.x = 0.0  # 회전 중 직진 금지
                else:
                    vel_msg.angular.z = 0.15*self.correct_Gaussian_error['weight']
                    vel_msg.linear.x = 0.0  # 회전 중 직진 금지

            self.pub.publish(vel_msg)
            return

        # ==============================
        if self.path_requested == False:
            if self.battery < 10.0 and self.is_charging == False and self.is_going_charging_station == False:
                self.turtlebot_stop()
                print('배터리 충전소 이동 시작')
                self.request_new_path('charge')
                self.path_requested = True  # 한 번만 요청하도록 설정
                return
            
            if self.is_charging:
                # 배터리 충전
                self.battery += 1.0
                self.battery = min(self.battery, 100.0)
                #  배터리가 충전 중이면서, 배터리 잔량이 50% 미만인 경우, 다른 명령 수행 불가능
                return
                    
            vel_msg = Twist()
            if self.is_to_move == False: 
                vel_msg.angular.z = 0.0
                vel_msg.linear.x = 0.0
            # 🚨 장애물이 감지되면 이동을 멈추고 새로운 경로 요청
            else:
                # if self.object_detected and (self.path_requested == False) and (self.object_detected_cnt >= 0):
                #     return 
                # else:
                # 현재 목표까지의 거리 계산
                distance = math.sqrt((self.goal.x - self.pose_x) ** 2 + (self.goal.y - self.pose_y) ** 2)
                # 목표 지점 도착 여부 확인
                if distance < 0.1:
                    # self.get_logger().info(f"목표 지점 {self.current_goal_idx} 도착. 잠시 정지합니다.")
                    # print(f"목표 지점 {self.current_goal_idx} 도착. 잠시 정지합니다.")
                    # 목표 지점 도착 후 1초 정지
                    self.turtlebot_stop()
                    self.current_goal_idx += 1
                    self.set_new_goal()

                    return

                # 목표 heading 계산
                target_heading = math.degrees(math.atan2(-(self.goal.x - self.pose_x), self.goal.y - self.pose_y))
                target_heading = (target_heading + 360) % 360  # 0~360도로 변환

                # 현재 heading과 목표 heading 비교 (최단 회전 경로 고려)
                if 0.0 <= self.heading <= 180.0:
                    if self.heading <= target_heading <= self.heading + 180.0: weight = -1 
                    else: weight = 1
                else:
                    if self.heading - 180.0 <= target_heading <= self.heading: weight = 1
                    else: weight = -1

                # 각도 차이 구하기
                Ang_1, Ang_2 = self.heading, target_heading
                if Ang_1 < Ang_2: Ang_1, Ang_2 = Ang_2, Ang_1
                # 각도 차이 구하기
                angle_diff = abs(Ang_1 - Ang_2)
                if abs(Ang_1 - (Ang_2 + 360.0)) < angle_diff:
                    angle_diff = abs(Ang_1 - (Ang_2 + 360.0))

                # 🔹 heading이 목표와 10도 이상 차이나면 회전
                if angle_diff > 15.0:

                    # 회전 속도를 angle_diff에 비례하도록 조정 (단, 최대 속도 제한)
                    if angle_diff <= 30.0:
                        vel_msg.angular.z = 0.15*weight
                        # print('각도 미세조정 시작 ==========')
                        # 각도 미세조정 시작
                        self.correct_Gaussian_error['status'] = True
                        self.correct_Gaussian_error['target_heading'] = target_heading
                        self.correct_Gaussian_error['weight'] = weight
                    elif angle_diff <= 60.0:
                        vel_msg.angular.z = 0.15*weight
                        # vel_msg.angular.z = min(0.1, 0.01*(angle_diff/4))*weight
                    elif angle_diff <= 90.0:
                        vel_msg.angular.z = 0.2*weight
                        # vel_msg.angular.z = max(0.15, 0.01*(angle_diff/5))*weight
                    elif angle_diff <= 120.0:
                        vel_msg.angular.z = 0.25*weight
                        # vel_msg.angular.z = max(0.18, 0.01*(angle_diff/6))*weight
                    else:
                        vel_msg.angular.z = 0.3*weight
                    vel_msg.linear.x = 0.0  # 회전 중 직진 금지
            
                    # =======================================================================

                else:
                    vel_msg.linear.x = 0.2
                    vel_msg.angular.z = 0.1 * weight * angle_diff / 10.0  # 직진 중 부드러운 조향
                    # vel_msg.angular.z = 0.0  # 직진 시 회전 없음

            self.pub.publish(vel_msg)
        
        # self.is_to_move == False
        else:
            self.turtlebot_stop()
            return 
        
    def turtlebot_stop(self):
        # self.get_logger().info("Turtlebot stopping")
        # print("=================정지================")
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
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import numpy as np
import time

from .config import params_map, PKG_PATH, MQTT_CONFIG
import paho.mqtt.client as mqtt
from std_msgs.msg import Int32

class Controller(Node):
    def __init__(self):
        super().__init__('controller')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 1)
        self.a_star_global_path_sub = self.create_subscription(Path, '/a_star_global_path', self.global_path_callback, 1)
        self.object_detected_sub = self.create_subscription(Int32, '/object_detected', self.object_callback, 1)
        self.order_id_sub = self.create_subscription(Int32, '/order_id', self.order_id_callback, 1)
        #self.move_order_sub = self.create_subscription(Bool, '/move_order', self.move_order_callback, 1)
        #self.move_order_pub = self.create_publisher(Bool, '/move_order', 1)
        #self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_request_pub = self.create_publisher(Point, '/request_new_path', 1) # 장애물 감지 시 새 경로 요청
        self.cmd_msg = Twist()

        # 현재 위치 및 방향
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.heading = 0.0  # LaserScan에서 계산

        # 배터리 잔량
        self.is_charging = False
        self.prior_pose = 0.0
        self.present_pose = 0.0
        self.battery = 100.0

        # 이동 타이머 설정
        self.timer = self.create_timer(0.3, self.move_to_destination)

        self.is_to_move = False
        #self.is_order = False

        # a_star를 통해 생성한 global_path
        self.global_path = [(-51.0,-51.0)]
        
        self.current_goal_idx = 0

        # 목표 지점 설정
        self.goal = Point()
        self.set_new_goal()
        self.object_detected = False
        self.path_requested = False
        self.object_angle = 0
        self.order_id = None

        # MQTT 설정 
        self.mqtt_client = mqtt.Client()
        self.mqtt_broker = MQTT_CONFIG["BROKER"]
        self.mqtt_port = MQTT_CONFIG["PORT"]
        self.mqtt_topic = "robot/log"

        #self.mqtt_client.on_connect = self.on_connect
        #self.mqtt_client.on_message = self.on_message
        self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
        self.mqtt_client.loop_start()


    def scan_callback(self, msg):
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
        left = [val for val in self.ranges[:20] if val < 2.0]
        right = [val for val in self.ranges[339:359] if val < 2.0]
        front = [val for val in  self.ranges[:10] if val < 2.0] + [val for val in self.ranges[349:359] if val < 2.0]
        left = sum(left) / len(left) if len(left) else 100
        right = sum(right) / len(right) if len(right) else 100
        front = sum(front) / len(front) if len(front) else 100
        pivot = min(front, right, left)


        if pivot < 0.8:
            if self.object_detected == False:
                self.object_detected = True
                print(pivot, 'pivot')
                if front == pivot: print('정면 장애물 감지')
                elif right == pivot: print('우측면 장애물 감지')
                elif left == pivot: print('좌측면 장애물 감지')

                self.turtlebot_stop() 
                self.request_new_path()
                self.path_requested = True  # 한 번만 요청하도록 설정
        else:
            self.object_detected = False


    def order_id_callback(self, msg):
        self.order_id = msg.data
        print(f"명령 ID 수신: {self.order_id}")

    def global_path_callback(self, msg):
        path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.global_path = path
        self.goal.x = path[0][0]
        self.goal.y = path[0][1]
        # print(self.global_path)
        print("경로 받기 성공")
        self.current_goal_idx = 0
        self.is_to_move = True
        self.path_requested = False


    # def object_callback(self, msg):
    #     if msg.data:  # 장애물 감지됨
    #         if not self.object_detected: 
    #             print("🚨 장애물 감지! 이동 중단 및 경로 재설정 준비")

    #         self.object_detected = True
    #         self.object_angle = msg.data + self.heading
    #     else:
    #         if self.object_detected:
    #             print("✅ 장애물 해제됨, 이동 재개 가능")
    #         self.object_detected = False
    #         self.path_requested = False  # 장애물이 사라졌으니 다시 경로 재요청 가능

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
        print(self.current_goal_idx, ' 인덱스')
        #print(self.global_path[self.current_goal_idx], '다음좌표')
        """ 현재 목표를 리스트에서 설정 """
        if self.current_goal_idx < len(self.global_path):
            self.goal.x, self.goal.y = self.global_path[self.current_goal_idx]
        else:
            self.turtlebot_stop()
            self.get_logger().info("finish =========")
            self.is_to_move = False
            self.current_goal_idx = 0
            # rclpy.shutdown()

            payload = {
                "id": self.order_id if self.order_id is not None else -1,
                "status": "arrived"
            }
            self.mqtt_client.publish(self.mqtt_topic, json.dumps(payload))
            self.order_id = None

    def move_to_destination(self):
        print(f'배터리 잔량 {round(self.battery, 2)}%')
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

                # 현재 heading과 목표 heading 비교 (최단 회전 경로 고려)
                angle_diff = (target_heading - self.heading + 540) % 360 - 180
                print(target_heading, self.heading)
                # 🔹 heading이 목표와 5도 이상 차이나면 회전
                if abs(angle_diff) > 10:
                    print("heading이 목표와 5도 이상 차이나면 회전")
                    kp_angular = 0.01  # 회전 속도 조절 계수 (값을 더 키워도 됨)
                    max_angular_speed = 0.2  # 최대 회전 속도 제한

                    # # 회전 속도를 angle_diff에 비례하도록 조정 (단, 최대 속도 제한)
                    vel_msg.angular.z = -max(min(kp_angular * angle_diff, max_angular_speed), -max_angular_speed)
                    vel_msg.linear.x = 0.0  # 회전 중 직진 금지
            
                    # =======================================================================

                else:
                    print("heading 차이가 5도 이하라면 직진")
                    # 🔹 heading 차이가 5도 이하라면 직진
                    vel_msg.linear.x = max(0.2, min(distance, 1.5))  # 최대 속도 1.5
                    vel_msg.angular.z = 0.0  # 직진 시 회전 없음

        self.pub.publish(vel_msg)

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

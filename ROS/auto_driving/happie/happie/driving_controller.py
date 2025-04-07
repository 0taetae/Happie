import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import time

from .config import params_map, PKG_PATH, MQTT_CONFIG
import paho.mqtt.client as mqtt

class Controller(Node):
    def __init__(self):
        super().__init__('controller')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 1)
        self.a_star_global_path_sub = self.create_subscription(Path, '/a_star_global_path', self.global_path_callback, 1)
        #self.move_order_sub = self.create_subscription(Bool, '/move_order', self.move_order_callback, 1)
        #self.move_order_pub = self.create_publisher(Bool, '/move_order', 1)
        #self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.cmd_msg = Twist()

        # 현재 위치 및 방향
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.heading = 0.0  # LaserScan에서 계산

        # 이동 타이머 설정
        self.timer = self.create_timer(0.1, self.move_to_destination)

        self.is_to_move = False
        #self.is_order = False

        # a_star를 통해 생성한 global_path
        self.global_path = [(-1.0,-1.0)]
        self.current_goal_idx = 0

        # 목표 지점 설정
        self.goal = Point()
        self.set_new_goal()

        # MQTT 설정 
        self.mqtt_client = mqtt.Client()
        self.mqtt_broker = MQTT_CONFIG["BROKER"]
        self.mqtt_port = MQTT_CONFIG["PORT"]
        self.mqtt_topic = "robot/log"

        #self.mqtt_client.on_connect = self.on_connect
        #self.mqtt_client.on_message = self.on_message
        self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
        self.mqtt_client.loop_start()
    
    #def move_order_callback(self, msg):
        # 중복 처리 방지
        # if self.is_order == msg.data:
        #     return

        # self.is_order = msg.data
        # if msg.data:
        #     self.get_logger().info("Received move_order: True")
        # else:
        #     self.get_logger().info("Received move_order: False")
        #     self.is_to_move = False  # 명시적으로 멈춤
    
    def scan_callback(self, msg):
        # LaserScan 데이터를 받아 현재 위치와 heading 업데이트 
        self.pose_x = msg.range_min
        self.pose_y = msg.scan_time 

        # heading 값 계산 (예제, 실제 데이터에서 계산 필요)
        self.heading = (msg.time_increment + 360) % 360
        print(f"현재 위치: ({round(self.pose_x, 3)}, {round(self.pose_y, 3)})")
        #print(f"현재 heading: {round(self.heading, 2)}°")

    def global_path_callback(self, msg):
        #if self.is_order:
        path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.global_path = path
        self.goal.x = path[0][0]
        self.goal.y = path[0][1]
        print(self.global_path)
        print("경로 받기 성공")
        self.current_goal_idx = 0
        self.is_to_move = True
            

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

            self.mqtt_client.publish(self.mqtt_topic, "arrived")

    def move_to_destination(self):
        if self.is_to_move == False: return 

        vel_msg = Twist()
        # 현재 목표까지의 거리 계산
        distance = math.sqrt((self.goal.x - self.pose_x) ** 2 + (self.goal.y - self.pose_y) ** 2)
        print(distance,'distance')
        # 목표 지점 도착 여부 확인
        if distance < 0.1:
            # self.get_logger().info(f"목표 지점 {self.current_goal_idx} 도착. 잠시 정지합니다.")
            print(f"목표 지점 {self.current_goal_idx} 도착. 잠시 정지합니다.")
            print(self.is_to_move)
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

        # 🔹 heading이 목표와 5도 이상 차이나면 회전
        if abs(angle_diff) > 5:
            print("heading이 목표와 5도 이상 차이나면 회전")
            kp_angular = 0.02  # 회전 속도 조절 계수 (값을 더 키워도 됨)
            max_angular_speed = 1.0  # 최대 회전 속도 제한

            # 회전 속도를 angle_diff에 비례하도록 조정 (단, 최대 속도 제한)
            vel_msg.angular.z = -max(min(kp_angular * angle_diff, max_angular_speed), -max_angular_speed)
            vel_msg.linear.x = 0.0  # 회전 중 직진 금지
            # print(f'현재 heading: {self.heading}')
            # print(f'현재 각속도: {vel_msg.angular.z}')

        else:
            print("heading 차이가 5도 이하라면 직진")
            # 🔹 heading 차이가 5도 이하라면 직진
            kp_linear = 1  # 이동 속도 조절 계수
            vel_msg.linear.x = min(kp_linear * distance, 0.7)  # 최대 속도 0.5
            vel_msg.angular.z = 0.0  # 직진 시 회전 없음

        # 디버깅 출력

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

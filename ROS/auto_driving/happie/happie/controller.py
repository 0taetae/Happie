import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from ssafy_msgs.msg import TurtlebotStatus, EnviromentStatus
from std_msgs.msg import Int8MultiArray
from math import sqrt
from geometry_msgs.msg import PoseStamped 
from sensor_msgs.msg import LaserScan

import paho.mqtt.client as mqtt  # MQTT 라이브러리 추가
import json  # 데이터를 JSON으로 변환하기 위함

class Controller(Node):

    def __init__(self):
        super().__init__('controller')
        
        self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.app_control_pub = self.create_publisher(Int8MultiArray, 'app_control', 10)

        self.turtlebot_status_sub = self.create_subscription(TurtlebotStatus, '/turtlebot_status', self.listener_callback, 10)
        self.envir_status_sub = self.create_subscription(EnviromentStatus, '/envir_status', self.envir_callback, 10)
        self.app_status_sub = self.create_subscription(Int8MultiArray, '/app_status', self.app_callback, 10)
        self.timer = self.create_timer(0.033, self.timer_callback)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        

        self.cmd_msg = Twist()
        
        self.app_control_msg = Int8MultiArray()
        self.app_control_msg.data = [0] * 17

        self.turtlebot_status_msg = TurtlebotStatus()
        self.envir_status_msg = EnviromentStatus()
        self.app_status_msg = Int8MultiArray()
        self.is_turtlebot_status = False
        self.is_app_status = False
        self.is_envir_status = False

        # MQTT Client 설정
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message

        self.pose_x = 0.0  # 로봇의 x 좌표
        self.pose_y = 0.0  # 로봇의 y 좌표

        # MQTT 메시지를 통해 x,y 좌표를 받음 
        self.mqtt_broker = "j12e103.p.ssafy.io"
        self.mqtt_port = 1883
        self.mqtt_topic = "robot/destination"
        self.mqtt_username = "happie_mqtt_user"
        self.mqtt_password = "gkstkfckdl0411!"
        
        self.mqtt_client.username_pw_set(self.mqtt_username, self.mqtt_password)
        self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
        self.mqtt_client.subscribe(self.mqtt_topic)  # 목적지 좌표 수신
        self.mqtt_client.loop_start()

        self.destination_x = 0.0
        self.destination_y = 0.0

    # MQTT 연결 완료 시 호출되는 콜백 함수
    def on_connect(self, client, userdata, flags, rc):
        self.get_logger().info(f"Connected to MQTT broker with result code {rc}")

    # MQTT 메시지 수신 시 호출되는 콜백 함수
    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            self.destination_x = payload.get('x', 0.0)  # x 좌표
            self.destination_y = payload.get('y', 0.0)  # y 좌표
            
            self.get_logger().info(f"Received destination: x={self.destination_x}, y={self.destination_y}")
        except json.JSONDecodeError:
            self.get_logger().error("Failed to decode MQTT message")


    def scan_callback(self, msg):
        self.pose_x = msg.range_min  # range_min을 x 좌표로 설정
        self.pose_y = msg.scan_time  # scan_time을 y 좌표로 설정

        self.get_logger().info(f"Updated pose from /scan: x={self.pose_x}, y={self.pose_y}")

        #destination_x = -5.3  # 임의로 설정한 x 좌표
        #destination_y = -6.3  # 임의로 설정한 y 좌표
        #self.move_to_destination(destination_x, destination_y)
        self.move_to_destination(self.destination_x, self.destination_y)


    # goal_pose를 퍼블리시
    def move_to_destination(self, x, y):

        
        self.get_logger().info(f"Moving to destination: ({x}, {y})")

        # 로봇의 현재 위치 
        current_x = self.pose_x   # scan_callback에서 받은 range_min을 x 좌표로 사용
        current_y = self.pose_y   # scan_callback에서 받은 scan_time을 y 좌표로 사용
        print(current_x)
        print(current_y)


        # 목표 위치와 현재 위치의 차이
        distance = sqrt((x - current_x)**2 + (y - current_y)**2)
        print("------------")
        print(distance)
        # 목표 위치에 근접하면 멈추기
        if distance < 0.5:
            self.turtlebot_stop()
            #self.get_logger().info("Destination reached, stopping the robot.")
            print("멈추기")
        else:
            print("직진")
            #self.cmd_msg.linear.x = 0.5  # 직진 속도
            #self.cmd_msg.angular.z = 0.0  # 회전하지 않음
            #self.cmd_publisher.publish(self.cmd_msg)
            self.turtlebot_go() 

    def listener_callback(self, msg):
        self.is_turtlebot_status = True
        self.turtlebot_status_msg = msg

    def envir_callback(self, msg):
        self.is_envir_status = True
        self.envir_status_msg = msg

    def app_callback(self, msg):
        self.is_app_status = True
        self.app_status_msg = msg  

    def app_all_on(self):
        # self.get_logger().info("Turning all appliances ON")
        self.app_control_msg.data = [1] * 17
        self.app_control_pub.publish(self.app_control_msg)
        
    def app_all_off(self):
        # self.get_logger().info("Turning all appliances OFF")
        self.app_control_msg.data = [2] * 17
        self.app_control_pub.publish(self.app_control_msg)
        
    def app_on_select(self, num):
        # self.get_logger().info(f"Turning appliance {num} ON")
        self.app_control_msg.data[num] = 1
        self.app_control_pub.publish(self.app_control_msg)

    def app_off_select(self, num):
        # self.get_logger().info(f"Turning appliance {num} OFF")
        self.app_control_msg.data[num] = 2
        self.app_control_pub.publish(self.app_control_msg)

    def turtlebot_go(self):
        #self.get_logger().info("Turtlebot moving forward")
        self.get_logger().info("Turtlebot gogogo")
        self.cmd_msg.linear.x = 1.0
        self.cmd_msg.angular.z = 0.0
        self.cmd_publisher.publish(self.cmd_msg)

    def turtlebot_stop(self):
        # self.get_logger().info("Turtlebot stopping")
        self.cmd_msg.linear.x = 0.0
        self.cmd_msg.angular.z = 0.0

    def turtlebot_cw_rot(self):
        # self.get_logger().info("Turtlebot rotating clockwise")
        self.cmd_msg.linear.x = 0.0
        self.cmd_msg.angular.z = -0.5

    def turtlebot_cww_rot(self):
        # self.get_logger().info("Turtlebot rotating counter-clockwise")
        self.cmd_msg.linear.x = 0.0
        self.cmd_msg.angular.z = 0.5

    def timer_callback(self):
        # self.turtlebot_cw_rot()
        #self.turtlebot_go()

        # if self.is_turtlebot_status:
        #     # self.get_logger().info(f"Turtlebot Status: Linear Vel: {self.turtlebot_status_msg.twist.linear.x}, "
        #     #                        f"Angular Vel: {self.turtlebot_status_msg.twist.angular.z}, "
        #     #                        f"Battery Percentage: {self.turtlebot_status_msg.battery_percentage}, "
        #     #                        f"Power Supply Status: {self.turtlebot_status_msg.power_supply_status}")
            
        # if self.is_envir_status:
        #     # self.get_logger().info(f"Environment Status: {self.envir_status_msg.month}/{self.envir_status_msg.day} "
        #     #            f"{self.envir_status_msg.hour}:{self.envir_status_msg.minute}, "
        #     #            f"Temperature: {self.envir_status_msg.temperature}°C, Weather: {self.envir_status_msg.weather}")
        
        # if self.is_app_status:
        #     # self.get_logger().info(f"Appliance Status: {self.app_status_msg.data}")
        
        self.cmd_publisher.publish(self.cmd_msg)


def main(args=None):
    rclpy.init(args=args)
    sub1_controller = Controller()
    rclpy.spin(sub1_controller)
    sub1_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

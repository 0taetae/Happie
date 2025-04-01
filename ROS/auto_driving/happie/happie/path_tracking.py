import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, Point
from ssafy_msgs.msg import TurtlebotStatus
from squaternion import Quaternion
from nav_msgs.msg import Odometry, Path
from math import pi, cos, sin, sqrt, atan2
import numpy as np

class followTheCarrot(Node):

    def __init__(self):
        super().__init__('path_tracking')
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.status_sub = self.create_subscription(TurtlebotStatus, '/turtlebot_status', self.status_callback, 10)
        self.path_sub = self.create_subscription(Path, '/local_path', self.path_callback, 10)

        # 로직 1. 제어 주기 및 타이머 설정
        time_period = 0.05
        self.timer = self.create_timer(time_period, self.timer_callback)

        self.is_odom = False
        self.is_path = False
        self.is_status = False

        self.odom_msg = Odometry()
        self.robot_yaw = 0.0
        self.path_msg = Path()
        self.cmd_msg = Twist()

        # 로직 2. 파라미터 설정
        self.lfd = 0.1
        self.min_lfd = 0.1
        self.max_lfd = 1.0

    def timer_callback(self):
        if self.is_status and self.is_odom and self.is_path:
            if len(self.path_msg.poses) > 1:
                self.is_look_forward_point = False
                
                robot_pose_x = self.odom_msg.pose.pose.position.x
                robot_pose_y = self.odom_msg.pose.pose.position.y

                # 로봇이 경로로부터 떨어진 거리 계산
                closest_point = self.path_msg.poses[0].pose.position
                lateral_error = sqrt(pow(closest_point.x - robot_pose_x, 2) +
                                     pow(closest_point.y - robot_pose_y, 2))
                self.get_logger().info(f"Lateral Error (Distance to Path): {lateral_error:.4f} m")
                
                # 로직 4. 전방 주시거리 설정
                self.lfd = max(self.min_lfd, min(self.max_lfd, lateral_error))

                min_dis = float('inf')
                
                # 로직 5. 전방 주시 포인트 설정
                for num, waypoint in enumerate(self.path_msg.poses):
                    current_point = waypoint.pose.position
                    dis = sqrt(pow(current_point.x - robot_pose_x, 2) + pow(current_point.y - robot_pose_y, 2))
                    if abs(dis - self.lfd) < min_dis:
                        min_dis = abs(dis - self.lfd)
                        self.forward_point = current_point
                        self.is_look_forward_point = True
                
                if self.is_look_forward_point:
                    global_forward_point = [self.forward_point.x, self.forward_point.y, 1]
                    
                    # 로직 6. 전방 주시 포인트와 로봇 헤딩과의 각도 계산
                    trans_matrix = np.array([[cos(self.robot_yaw), -sin(self.robot_yaw), robot_pose_x],
                                             [sin(self.robot_yaw), cos(self.robot_yaw), robot_pose_y],
                                             [0, 0, 1]])
                    
                    det_trans_matrix = np.linalg.inv(trans_matrix)
                    local_forward_point = np.dot(det_trans_matrix, global_forward_point)
                    theta = atan2(local_forward_point[1], local_forward_point[0])
                    
                    # 로직 7. 선속도, 각속도 정하기
                    out_vel = max(0.1, 1.0 - abs(theta))
                    out_rad_vel = theta * 2.0
                    
                    self.cmd_msg.linear.x = out_vel
                    self.cmd_msg.angular.z = out_rad_vel
                else:
                    self.cmd_msg.linear.x = 0.0
                    self.cmd_msg.angular.z = 0.0
            else:
                self.cmd_msg.linear.x = 0.0
                self.cmd_msg.angular.z = 0.0

            self.cmd_pub.publish(self.cmd_msg)

    def odom_callback(self, msg):
        self.is_odom = True
        self.odom_msg = msg
        
        # 로직 3. Quaternion 을 euler angle 로 변환
        q = msg.pose.pose.orientation
        _, _, self.robot_yaw = Quaternion(q.w, q.x, q.y, q.z).to_euler()

        # 로봇 위치 출력
        self.get_logger().info(f"Robot Position: x={self.odom_msg.pose.pose.position.x:.4f}, y={self.odom_msg.pose.pose.position.y:.4f}")

    def path_callback(self, msg):
        self.is_path = True
        self.path_msg = msg

    def status_callback(self, msg):
        self.is_status = True
        self.status_msg = msg


def main(args=None):
    rclpy.init(args=args)
    path_tracker = followTheCarrot()
    rclpy.spin(path_tracker)
    path_tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


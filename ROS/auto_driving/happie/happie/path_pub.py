import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist,PoseStamped
from squaternion import Quaternion
from nav_msgs.msg import Odometry,Path

from math import pi,cos,sin,sqrt
import tf2_ros
import os

from .config import params_map, PKG_PATH

class pathPub(Node):

    def __init__(self):
        super().__init__('path_pub')

        # 로직 1. publisher, subscriber 만들기
        self.global_path_pub = self.create_publisher(Path, 'global_path', 10)
        self.local_path_pub = self.create_publisher(Path, 'local_path', 10)
        self.subscription = self.create_subscription(Odometry, '/odom', self.listener_callback, 10)

        self.odom_msg = Odometry()
        self.is_odom = False

        # 전역경로 메시지
        self.global_path_msg = Path()
        self.global_path_msg.header.frame_id = 'map'

        # 로직 2. 만들어 놓은 경로 데이터를 읽기 모드로 open
        full_path = os.path.join(PKG_PATH, "path.txt")

        # 로직 3. 경로 데이터를 읽어서 Path 메시지에 데이터를 넣기
        lines = self.f.readlines()
        for line in lines:
            tmp = line.strip().split('\t')  # 탭을 기준으로 분리
            read_pose = PoseStamped()
            read_pose.pose.position.x = float(tmp[0])
            read_pose.pose.position.y = float(tmp[1])
            read_pose.pose.orientation.w = 1.0
            self.global_path_msg.poses.append(read_pose)
        
        self.f.close()

        # 로직 4. 주기마다 실행되는 타이머함수 생성, local_path_size 설정
        time_period = 0.02
        self.timer = self.create_timer(time_period, self.timer_callback)
        self.local_path_size = 20
        self.count = 0

    def listener_callback(self, msg):
        """ /odom 데이터 수신 시 실행되는 콜백 함수 """
        self.is_odom = True
        self.odom_msg = msg
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.get_logger().info(f"Received Odom: x={self.robot_x}, y={self.robot_y}")

    def timer_callback(self):
        if self.is_odom:
            local_path_msg = Path()
            local_path_msg.header.frame_id = 'map'
            
            x = self.odom_msg.pose.pose.position.x
            y = self.odom_msg.pose.pose.position.y
            current_waypoint = -1

            # 로봇 현재 위치 출력
            self.get_logger().info(f"Current Robot Position: x={x}, y={y}")

            # 로직 5. global_path 중 로봇과 가장 가까운 포인트 계산
            min_dis = float('inf')
            for i, waypoint in enumerate(self.global_path_msg.poses):
                distance = ((waypoint.pose.position.x - x) ** 2 + (waypoint.pose.position.y - y) ** 2) ** 0.5
                if distance < min_dis:
                    min_dis = distance
                    current_waypoint = i

            # 로직 6. local_path 예외 처리
            if current_waypoint != -1:
                if current_waypoint + self.local_path_size < len(self.global_path_msg.poses):
                    local_path_msg.poses = self.global_path_msg.poses[current_waypoint:current_waypoint + self.local_path_size]
                else:
                    local_path_msg.poses = self.global_path_msg.poses[current_waypoint:]
            
            self.local_path_pub.publish(local_path_msg)
            self.get_logger().info(f"Published local_path with {len(local_path_msg.poses)} poses")

        # 로직 7. global_path 업데이트 주기 재설정
        if self.count % 10 == 0:
            self.global_path_pub.publish(self.global_path_msg)
        self.count += 1


def main(args=None):
    rclpy.init(args=args)
    path_pub = pathPub()
    rclpy.spin(path_pub)
    path_pub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

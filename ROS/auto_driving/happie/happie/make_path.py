import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from squaternion import Quaternion
from nav_msgs.msg import Odometry,Path
import os
from math import sqrt

from .config import params_map, PKG_PATH

class makePath(Node):
    def __init__(self):
        super().__init__('make_path')

        # 로직 1. 노드에 필요한 publisher, subscriber 생성      
        self.path_pub = self.create_publisher(Path, 'global_path', 10)
        self.subscription = self.create_subscription(Odometry, '/odom', self.listener_callback, 10)

        # 로직 2. 저장할 경로 및 텍스트파일 이름을 정하고, 쓰기 모드로 열기
        os.makedirs(PKG_PATH, exist_ok=True)
        full_path = os.path.join(PKG_PATH, 'path.txt')
        self.f = open(full_path, 'w')  # 쓰기 모드로 열기
        
        self.is_odom = False  # 초기 상태를 False로 설정
        self.prev_x = 0.0
        self.prev_y = 0.0

        self.path_msg = Path()
        self.path_msg.header.frame_id = 'map'

    def listener_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        print(f'x : {x}, y : {y}')
        
        if not self.is_odom:
            # 로직 3. 콜백함수에서 처음 메시지가 들어오면 초기 위치를 저장
            self.is_odom = True
            self.prev_x = x
            self.prev_y = y
        else:
            # 로직 4. 콜백함수에서 이전 위치와 현재 위치의 거리 계산
            distance = sqrt((x - self.prev_x) ** 2 + (y - self.prev_y) ** 2)
            
            if distance > 0.1:
                # 로직 5. 거리 차이가 0.1m 이상이면 path_msg.poses에 추가하고 publish
                waypoint_pose = PoseStamped()
                waypoint_pose.pose.position.x = x
                waypoint_pose.pose.position.y = y
                waypoint_pose.pose.orientation.w = 1.0
                self.path_msg.poses.append(waypoint_pose)
                self.path_pub.publish(self.path_msg)
                
                # 로직 6. x,y 를 문자열로 변환하여 /t 로 구분 후 저장
                data = f'{x}\t{y}\n'
                self.f.write(data)
                self.prev_x = x
                self.prev_y = y


def main(args=None):
    rclpy.init(args=args)
    odom_based_make_path = makePath()
    rclpy.spin(odom_based_make_path)
    odom_based_make_path.f.close()
    odom_based_make_path.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

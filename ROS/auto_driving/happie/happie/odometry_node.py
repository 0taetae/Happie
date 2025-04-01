import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from ssafy_msgs.msg import TurtlebotStatus
from sensor_msgs.msg import Imu
from squaternion import Quaternion
from nav_msgs.msg import Odometry
from math import pi, cos, sin
import tf2_ros
import geometry_msgs.msg
import time


class Odom(Node):

    def __init__(self):
        super().__init__('odom')
        
        # 로직 1: publisher, subscriber, broadcaster 만들기
        self.subscription = self.create_subscription(TurtlebotStatus, '/turtlebot_status', self.listener_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.odom_publisher = self.create_publisher(Odometry, 'odom', 10)
        self.broadcaster = tf2_ros.TransformBroadcaster(self) ## TF를 사용하여 변환을 브로드캐스트함 

        # 로봇의 pose를 저장할 변수
        self.odom_msg = Odometry()
        self.base_link_transform = geometry_msgs.msg.TransformStamped()
        self.laser_transform = geometry_msgs.msg.TransformStamped()

        self.is_status = False
        self.is_imu = False
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.imu_offset = 0
        self.prev_time = 0

        # 로직 2: 메시지 설정
        self.odom_msg.header.frame_id = 'map'
        self.odom_msg.child_frame_id = 'base_link'

        self.base_link_transform.header.frame_id = 'map'
        self.base_link_transform.child_frame_id = 'base_link'

        self.laser_transform.header.frame_id = 'base_link'
        self.laser_transform.child_frame_id = 'laser'
        self.laser_transform.transform.translation.x = 0.2  # 예제 값
        self.laser_transform.transform.translation.y = 0.0
        self.laser_transform.transform.translation.z = 0.1
        self.laser_transform.transform.rotation.w = 1.0

    def imu_callback(self, msg):
        # 로직 3: IMU 에서 받은 quaternion을 euler angle로 변환
        imu_q = Quaternion(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z)
        roll, pitch, yaw = imu_q.to_euler()

        if not self.is_imu:
            self.is_imu = True
            self.imu_offset = yaw
        else:
            self.theta = yaw - self.imu_offset

    def listener_callback(self, msg):
        print(f'linear_vel: {msg.twist.linear.x}, angular_vel: {-msg.twist.angular.z}')

        if self.is_imu:
            if not self.is_status:
                self.is_status = True
                self.prev_time = self.get_clock().now()
            else:
                current_time = self.get_clock().now()
                period = (current_time - self.prev_time).nanoseconds / 1e9

                linear_x = msg.twist.linear.x
                angular_z = -msg.twist.angular.z

                # 로직 4: 로봇 위치 추정
                self.x += linear_x * cos(self.theta) * period
                self.y += linear_x * sin(self.theta) * period
                self.theta += angular_z * period

                self.base_link_transform.header.stamp = current_time.to_msg()
                self.laser_transform.header.stamp = current_time.to_msg()

                # 로직 5: 추정한 로봇 위치를 메시지에 담아 publish, broadcast
                q = Quaternion.from_euler(0, 0, self.theta)

                self.base_link_transform.transform.translation.x = self.x
                self.base_link_transform.transform.translation.y = self.y
                self.base_link_transform.transform.rotation.x = q.x
                self.base_link_transform.transform.rotation.y = q.y
                self.base_link_transform.transform.rotation.z = q.z
                self.base_link_transform.transform.rotation.w = q.w

                self.odom_msg.pose.pose.position.x = self.x
                self.odom_msg.pose.pose.position.y = self.y
                self.odom_msg.pose.pose.orientation.x = q.x
                self.odom_msg.pose.pose.orientation.y = q.y
                self.odom_msg.pose.pose.orientation.z = q.z
                self.odom_msg.pose.pose.orientation.w = q.w
                self.odom_msg.twist.twist.linear.x = linear_x
                self.odom_msg.twist.twist.angular.z = angular_z

                self.broadcaster.sendTransform(self.base_link_transform)
                self.broadcaster.sendTransform(self.laser_transform)
                self.odom_publisher.publish(self.odom_msg)

                self.prev_time = current_time


def main(args=None):
    rclpy.init(args=args)
    sub1_odom = Odom()
    rclpy.spin(sub1_odom)
    sub1_odom.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class LidarSubscriber(Node):
    def __init__(self):
        super().__init__('lidar_subscriber')
        ## LiDAR 센서의 데이터를 수신하기 위한 subscriber 생성 
        self.subscription = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10) ## 주제, 메시지 타입, 콜백함수, 큐사이즈(수신 버퍼 크기)
        self.subscription

    def lidar_callback(self, msg):
        self.get_logger().info(f'LiDAR Data: {msg.ranges[:10]}')  # 앞쪽 10개 거리 값 출력

def main(args=None):
    rclpy.init(args=args)
    node = LidarSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
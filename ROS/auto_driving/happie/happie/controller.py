import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from ssafy_msgs.msg import TurtlebotStatus, EnviromentStatus
from std_msgs.msg import Int8MultiArray

class Controller(Node):

    def __init__(self):
        super().__init__('controller')
        
        self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.app_control_pub = self.create_publisher(Int8MultiArray, 'app_control', 10)

        self.turtlebot_status_sub = self.create_subscription(TurtlebotStatus, '/turtlebot_status', self.listener_callback, 10)
        self.envir_status_sub = self.create_subscription(EnviromentStatus, '/envir_status', self.envir_callback, 10)
        self.app_status_sub = self.create_subscription(Int8MultiArray, '/app_status', self.app_callback, 10)
        self.timer = self.create_timer(0.033, self.timer_callback)

        self.cmd_msg = Twist()
        
        self.app_control_msg = Int8MultiArray()
        self.app_control_msg.data = [0] * 17

        self.turtlebot_status_msg = TurtlebotStatus()
        self.envir_status_msg = EnviromentStatus()
        self.app_status_msg = Int8MultiArray()
        self.is_turtlebot_status = False
        self.is_app_status = False
        self.is_envir_status = False

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
        self.get_logger().info("Turning all appliances ON")
        self.app_control_msg.data = [1] * 17
        self.app_control_pub.publish(self.app_control_msg)
        
    def app_all_off(self):
        self.get_logger().info("Turning all appliances OFF")
        self.app_control_msg.data = [2] * 17
        self.app_control_pub.publish(self.app_control_msg)
        
    def app_on_select(self, num):
        self.get_logger().info(f"Turning appliance {num} ON")
        self.app_control_msg.data[num] = 1
        self.app_control_pub.publish(self.app_control_msg)

    def app_off_select(self, num):
        self.get_logger().info(f"Turning appliance {num} OFF")
        self.app_control_msg.data[num] = 2
        self.app_control_pub.publish(self.app_control_msg)

    def turtlebot_go(self):
        self.get_logger().info("Turtlebot moving forward")
        self.cmd_msg.linear.x = 1.0
        self.cmd_msg.angular.z = 0.0

    def turtlebot_stop(self):
        self.get_logger().info("Turtlebot stopping")
        self.cmd_msg.linear.x = 0.0
        self.cmd_msg.angular.z = 0.0

    def turtlebot_cw_rot(self):
        self.get_logger().info("Turtlebot rotating clockwise")
        self.cmd_msg.linear.x = 0.0
        self.cmd_msg.angular.z = -0.5

    def turtlebot_cww_rot(self):
        self.get_logger().info("Turtlebot rotating counter-clockwise")
        self.cmd_msg.linear.x = 0.0
        self.cmd_msg.angular.z = 0.5

    def timer_callback(self):
        # self.turtlebot_cw_rot()
        self.turtlebot_go()

        if self.is_turtlebot_status:
            self.get_logger().info(f"Turtlebot Status: Linear Vel: {self.turtlebot_status_msg.twist.linear.x}, "
                                   f"Angular Vel: {self.turtlebot_status_msg.twist.angular.z}, "
                                   f"Battery Percentage: {self.turtlebot_status_msg.battery_percentage}, "
                                   f"Power Supply Status: {self.turtlebot_status_msg.power_supply_status}")
            
        if self.is_envir_status:
            self.get_logger().info(f"Environment Status: {self.envir_status_msg.month}/{self.envir_status_msg.day} "
                       f"{self.envir_status_msg.hour}:{self.envir_status_msg.minute}, "
                       f"Temperature: {self.envir_status_msg.temperature}°C, Weather: {self.envir_status_msg.weather}")
        
        if self.is_app_status:
            self.get_logger().info(f"Appliance Status: {self.app_status_msg.data}")
        
        self.cmd_publisher.publish(self.cmd_msg)


def main(args=None):
    rclpy.init(args=args)
    sub1_controller = Controller()
    rclpy.spin(sub1_controller)
    sub1_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

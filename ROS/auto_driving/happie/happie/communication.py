import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from ssafy_msgs.msg import TurtlebotStatus

class Communicator(Node):

    def __init__(self):
        supper().__init('Communicator') ## 노드 이름 설정 
        self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel',10) ## 로봇의 속도 명령을 전송하기 위한 publisher 생성
        self.turtlebot_status_sub = self.create_subscription(TurtlebotStatus, '/turtlebot_status', self.status_callback,10) ## 로봇의 상태를 수신하기 위한 subscriber 생성 / 수신된 메시지를 처리하는 콜백함수는 self.status_callback 
        self.timer = self.create_timer(0.1, self.timer_callback) ## 주기적으로 실행되는 타이머 생성(0.1초마다 timer_callback 함수 호출)

        self.cme_msg=Twist() ## Twist 메시지 타입의 객체 생성 
        self.turtlebot_status_msg = TurtlebotStatus() ## TurtlebotStatus 메시지 타입의 객체 생성 
    
    ## 로봇의 상태메시지를 수신할 때마다 호출 
    def status_callback(self,msg):
        print('Linear velocity :{0} , Angular Velocity : {1} Battery : {2}'.format(msg.twist.linear.x,msg.twist.angular.z,msg.battery_percentage)) ## 선형 속도, 각속도, 배터리 잔량 출력 
    
    ## 타이머가 트리거될 때마다 호출 
    def timer_callback(self):
        self.cmd_msg.angular.z=1.0 ## 각속도를 1.0으로 설정 
        self.cmd_publisher.publish(self.cmd_msg) ## 이 명령을 publisher를 통해 전송 

## ROS2 환경을 초기화 
def main(args=None):
    rclpy.init(args=args)
    com = Communicator() ## Communicator 노드를 생성 
    rclpy.spin(com)
    com.destroy_node() 
    rclpy.shutdown()

## 이 스크립트가 직접 실행될 때 main() 함수 호출 
if __name__ == '__main__':
    main()
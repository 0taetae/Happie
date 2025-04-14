import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from ssafy_msgs.msg import TurtlebotStatus, HandControl

# HandControl 모드 상수 정의
NORMAL_MODE = 0
PREVIEW_MODE = 1
PICK_UP_MODE = 2
PUT_DOWN_MODE = 3

class HandControlNode(Node):
    def __init__(self):
        super().__init__('hand_control')
        
        # Publisher & Subscriber 생성
        self.hand_control_pub = self.create_publisher(HandControl, '/hand_control', 10) ## 로봇의 손 제어 명령을 전송하기 위한 publisher를 생성, 큐 사이즈 10
        self.turtlebot_status_sub = self.create_subscription(TurtlebotStatus, '/turtlebot_status', self.turtlebot_status_cb, 10) ## 로봇의 상태를 수신하기 위한 subscriber 생성 / 큐 사이즈는 10 / 수신된 메시지를 처리하는 콜백함수는 turtlebot_status_cb 
        self.hand_control_id_sub = self.create_subscription(Int32, '/hand_control_id', self.hand_control_callback, 10)
        self.timer = self.create_timer(1, self.timer_callback) ## 주기적으로 실행되는 타이머 생성 
        
        # 메시지 변수 생성
        self.hand_control_msg = HandControl()
        self.turtlebot_status_msg = TurtlebotStatus()
        self.is_turtlebot_status = False ## 로봇 상태 메시지가 수신되었는지 여부를 표시하는 플래그 
    
    ## 타이머가 트리거될 때마다 호출 : 사용자에게 메뉴를 선택하도록 요청, 선택한 메뉴에 따라 다양한 손 제어 명령을 수행 
    def timer_callback(self):
        print('Select Menu [0: status_check, 1: preview, 2: pick_up, 3: put_down]')
        menu = None        
        # menu = input('>> ')
        if menu == '0':               
            self.hand_control_status()
        elif menu == '1':
            self.hand_control_preview()               
        elif menu == '2':
            self.hand_control_pick_up()   
        elif menu == '3':
            self.hand_control_put_down()
        else:
            pass
    ## 로봇의 상태 출력 : 로봇의 배터리 잔량, 전원 공급 상태, 손 사용 가능 여부, 물건 놓기 가능 여부, 물건 들어올리기 가능 여부 표시 
    def hand_control_status(self):
        if self.is_turtlebot_status:
            print(f'Turtlebot Battery: {self.turtlebot_status_msg.battery_percentage}%')
            print(f'Power Supply Status: {self.turtlebot_status_msg.power_supply_status}')
            print(f'Can Use Hand: {self.turtlebot_status_msg.can_use_hand}')
            print(f'Can Put: {self.turtlebot_status_msg.can_put}')
            print(f'Can Lift: {self.turtlebot_status_msg.can_lift}')
        else:
            print('Turtlebot status not received yet.')
    
    ## 손 제어 미리보기 모드 
    def hand_control_preview(self):
        if not self.turtlebot_status_msg.can_use_hand:
            print('Hand control is not available!')
            return
        
        self.hand_control_msg.control_mode = PREVIEW_MODE
        self.hand_control_pub.publish(self.hand_control_msg)
        print('Preview mode activated.')
    
    ## 로봇이 물건을 들어올리도록 명령 
    def hand_control_pick_up(self):
        if not self.turtlebot_status_msg.can_lift:
            print('Cannot pick up! Robot is not in the right state.')
            return
        
        self.hand_control_msg.control_mode = PICK_UP_MODE
        self.hand_control_pub.publish(self.hand_control_msg)
        print('Pick up command sent.')
    
    ## 로봇이 물건을 놓도록 명령 
    def hand_control_put_down(self):        
        if not self.turtlebot_status_msg.can_put:
            print('Cannot put down! Robot is not in the right state.')
            return
        
        self.hand_control_msg.control_mode = PUT_DOWN_MODE
        self.hand_control_msg.put_distance = 0.3  # 적절한 거리 설정
        self.hand_control_msg.put_height = 0.2  # 적절한 높이 설정
        self.hand_control_pub.publish(self.hand_control_msg)
        print('Put down command sent.')
    
    ## 로봇 상태 메시지를 수신할 때마다 호출 
    def turtlebot_status_cb(self, msg):
        self.is_turtlebot_status = True
        self.turtlebot_status_msg = msg

    def hand_control_callback(self, msg: Int32):
        print(f'get hand control order: {msg}')
        if msg.data == 2:
            self.get_logger().info("Command 2 received: Pick up")
            self.hand_control_pick_up()
        elif msg.data == 3:
            self.get_logger().info("Command 3 received: Put down")
            self.hand_control_put_down()
        else:
            self.get_logger().warn(f"Unknown command received: {msg.data}")
        

def main(args=None):
    rclpy.init(args=args)
    hand_control_node = HandControlNode()
    rclpy.spin(hand_control_node)
    hand_control_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from collections import deque

# a_star 노드는  OccupancyGrid map을 받아 grid map 기반 최단경로 탐색 알고리즘을 통해 로봇이 목적지까지 가는 경로를 생성하는 노드입니다.
# 로봇의 위치(/pose), 맵(/map), 목표 위치(/goal_pose)를 받아서 전역경로(/global_path)를 만들어 줍니다. 
# goal_pose는 rviz2에서 2D Goal Pose 버튼을 누르고 위치를 찍으면 메시지가 publish 됩니다. 
# 주의할 점 : odom을 받아서 사용하는데 기존 odom 노드는 시작했을 때 로봇의 초기 위치가 x,y,heading(0,0,0) 입니다. 
#               로봇의 초기위치를 맵 상에서 로봇의 위치와 맞춰줘야 합니다. 
# 따라서 sub2의 odom 노드를 수정해줍니다. turtlebot_status 안에는 정답데이터(절대 위치)가 있는데 그 정보를 사용해서 맵과 로봇의 좌표를 맞춰 줍니다.


class a_star(Node):

    def __init__(self):
        super().__init__('a_Star')
        # 로직 1. publisher, subscriber 만들기
        self.map_sub = self.create_subscription(OccupancyGrid, 'map', self.map_callback, 1) # 맵 데이터 수신 
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 1) # 로봇 위치 수신 
        self.goal_sub = self.create_subscription(PoseStamped, 'goal_pose', self.goal_callback, 1) # 목표 위치 수신 
        self.a_star_pub = self.create_publisher(Path, 'global_path', 1) # 토픽을 발생하여 최단 경로 출력 
        
        self.map_msg=OccupancyGrid()
        self.odom_msg=Odometry()
        self.is_map=False
        self.is_odom=False
        self.is_found_path=False
        self.is_grid_update=False

        # 로직 2. 파라미터 설정
        self.goal = [184, 224] 
        self.map_size_x = 350
        self.map_size_y = 350
        self.map_resolution = 0.05
        # offset은 맵 좌표와 로봇 좌표를 맞추기 위한 값 
        self.map_offset_x = -16.75
        self.map_offset_y = -12.75
    
        self.GRIDSIZE = 350 
        # 8방향 탐색을 위한 이동 방향 벡터 
        self.dx = [-1, 0, 0, 1, -1, -1, 1, 1]
        self.dy = [0, 1, -1, 0, -1, 1, -1, 1]
        # 대각선 이동 시 유클리드 거리(1.414) 반영
        self.dCost = [1, 1, 1, 1, 1.414, 1.414, 1.414, 1.414]

    # 로직 3. 맵 데이터를 행렬로 바꾸기
    def grid_update(self):
        
        self.is_grid_update = True
        # map_msg.data는 1차원 리스트로 제공되므로 reshape을 사용하여 2차원 배열로 변경 
        self.grid = np.array(self.map_msg.data).reshape((self.map_size_y, self.map_size_x)) 

    # 실제 좌표를 맵의 그리드 셀 좌표로 변환 
    def pose_to_grid_cell(self, x, y):
        '''
        로직 4. 위치(x,y)를 map의 grid cell로 변환 
        (테스트) pose가 (-8,-4)라면 맵의 중앙에 위치하게 된다. 따라서 map_point_x,y 는 map size의 절반인 (175,175)가 된다.
        pose가 (-16.75,12.75) 라면 맵의 시작점에 위치하게 된다. 따라서 map_point_x,y는 (0,0)이 된다.
        '''

        map_point_x = int((x - self.map_offset_x) / self.map_resolution)
        map_point_y = int((y - self.map_offset_y) / self.map_resolution)
        return map_point_x, map_point_y

    # 그리드 셀 좌표를 실제 좌표로 변환 
    def grid_cell_to_pose(self, grid_cell):
        '''
        로직 5. map의 grid cell을 위치(x,y)로 변환
        (테스트) grid cell이 (175,175)라면 맵의 중앙에 위치하게 된다. 따라서 pose로 변환하게 되면 맵의 중앙인 (-8,-4)가 된다.
        grid cell이 (350,350)라면 맵의 제일 끝 좌측 상단에 위치하게 된다. 따라서 pose로 변환하게 되면 맵의 좌측 상단인 (0.75,6.25)가 된다.
        '''

        x = (grid_cell[0] * self.map_resolution) + self.map_offset_x
        y = (grid_cell[1] * self.map_resolution) + self.map_offset_y
        return [x, y]

    # 로봇의 위치 데이터 저장 
    def odom_callback(self, msg):
        self.is_odom = True
        self.odom_msg = msg

    # 맵 데이터를 저장 
    def map_callback(self, msg):
        self.is_map = True
        self.map_msg = msg

    # 목표 위치 설정 및 경로 탐색 
    def goal_callback(self, msg):
        if msg.header.frame_id == 'map':
            ## 로직 6. goal_pose 메시지 수신하여 목표 위치 설정
            goal_x = msg.pose.position.x
            goal_y = msg.pose.position.y
            goal_cell = self.pose_to_grid_cell(goal_x, goal_y)
            self.goal = goal_cell


            print(
            f"geometry_msgs.msg.PoseStamped(header=std_msgs.msg.Header("
            f"stamp=builtin_interfaces.msg.Time(sec={msg.header.stamp.sec}, "
            f"nanosec={msg.header.stamp.nanosec}), frame_id='{msg.header.frame_id}'), "
            f"pose=geometry_msgs.msg.Pose(position=geometry_msgs.msg.Point("
            f"x={goal_x:.6f}, y={goal_y:.6f}, z={msg.pose.position.z:.6f}), "
            f"orientation=geometry_msgs.msg.Quaternion("
            f"x={msg.pose.orientation.x:.6f}, y={msg.pose.orientation.y:.6f}, "
            f"z={msg.pose.orientation.z:.6f}, w={msg.pose.orientation.w:.6f})))")
            
            if self.is_map and self.is_odom:
                if not self.is_grid_update:
                    self.grid_update()

                self.final_path = []
                # 현재 로봇의 위치를 그리드 좌표로 변환 
                x = self.odom_msg.pose.pose.position.x
                y = self.odom_msg.pose.pose.position.y
                start_grid_cell = self.pose_to_grid_cell(x, y)
                
                self.path = np.zeros((self.GRIDSIZE, self.GRIDSIZE), dtype=int)
                self.cost = np.full((self.GRIDSIZE, self.GRIDSIZE), self.GRIDSIZE*self.GRIDSIZE)

                # 시작지, 목적지가 탐색가능한 영역이고, 시작지와 목적지가 같지 않으면 경로탐색
                if self.grid[start_grid_cell[0], start_grid_cell[1]] == 0 and \
                   self.grid[self.goal[0], self.goal[1]] == 0 and \
                   start_grid_cell != self.goal:
                    self.dijkstra(start_grid_cell)

                self.global_path_msg = Path()
                self.global_path_msg.header.frame_id = 'map'
                for grid_cell in reversed(self.final_path):
                    tmp_pose = PoseStamped()
                    waypoint_x, waypoint_y = self.grid_cell_to_pose(grid_cell)
                    tmp_pose.pose.position.x = waypoint_x
                    tmp_pose.pose.position.y = waypoint_y
                    tmp_pose.pose.orientation.w = 1.0
                    self.global_path_msg.poses.append(tmp_pose)
                
                if self.final_path:
                    self.a_star_pub.publish(self.global_path_msg)

    def dijkstra(self, start):
        Q = deque()
        Q.append(start)
        self.cost[start[0]][start[1]] = 1
        found = False

        ## 로직 7. grid 기반 최단경로 탐색
        while Q:
            current = Q.popleft()
            if current == self.goal:
                found = True
                break

            for i in range(8):
                next_node = (current[0] + self.dx[i], current[1] + self.dy[i])
                if 0 <= next_node[0] < self.GRIDSIZE and 0 <= next_node[1] < self.GRIDSIZE:
                    if self.grid[next_node[0], next_node[1]] < 50:
                        if self.cost[next_node[0], next_node[1]] > self.cost[current[0], current[1]] + self.dCost[i]:
                            Q.append(next_node)
                            self.path[next_node[0]][next_node[1]] = current
                            self.cost[next_node[0]][next_node[1]] = self.cost[current[0]][current[1]] + self.dCost[i]
        
        if found:
            node = self.goal
            while node != start:
                self.final_path.append(node)
                node = self.path[node[0]][node[1]]
            self.final_path.append(start)
            self.final_path.reverse()
        

def main(args=None):
    rclpy.init(args=args)
    global_planner = a_star()
    rclpy.spin(global_planner)
    global_planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

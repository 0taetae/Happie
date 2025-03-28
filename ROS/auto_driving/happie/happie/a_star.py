import rclpy
import numpy as np
from rclpy.node import Node
import os
from geometry_msgs.msg import Pose, PoseStamped, Point
from squaternion import Quaternion
from nav_msgs.msg import Odometry,OccupancyGrid,MapMetaData,Path
from math import pi,cos,sin,sqrt
import heapq
from std_msgs.msg import String

from config import params_map, PKG_PATH

# a_star 노드는  OccupancyGrid map을 받아 grid map 기반 최단경로 탐색 알고리즘을 통해 로봇이 목적지까지 가는 경로를 생성하는 노드입니다.

# 노드 로직 순서
# 1. publisher, subscriber 만들기
# 2. 파라미터 설정
# 3. 맵 데이터 행렬로 바꾸기
# 4. 위치(x,y)를 map의 grid cell로 변환
# 5. map의 grid cell을 위치(x,y)로 변환
# 6. goal_pose 메시지 수신하여 목표 위치 설정
# 7. grid 기반 최단경로 탐색

class a_star(Node):

    def __init__(self):
        super().__init__('a_star')
        # 로직 1. publisher, subscriber 만들기
        self.map_sub = self.create_subscription(OccupancyGrid,'map',self.map_callback,1)
        self.odom_sub = self.create_subscription(Odometry,'odom',self.odom_callback,1)
        self.obs_sub = self.create_subscription(String, 'obs_msg', self.obs_callback, 1)
        self.a_star_goal_sub = self.create_subscription(Point,'a_star_goal',self.a_star_goal_callback,1)
        self.a_star_pub= self.create_publisher(Path, 'global_path', 1)
        time_period=0.1
        self.timer = self.create_timer(time_period, self.timer_callback)

        self.map_msg=OccupancyGrid()
        self.odom_msg=Odometry()
        self.is_map=False
        self.is_odom=False
        self.is_found_path=False
        self.is_grid_update=False


        # 로직 2. 파라미터 설정
        self.goal = [184,224]
        self.map_size_x = int(params_map["MAP_SIZE"][0] / params_map["MAP_RESOLUTION"])
        self.map_size_y = int(params_map["MAP_SIZE"][1] / params_map["MAP_RESOLUTION"])
        self.map_resolution = params_map["MAP_RESOLUTION"]

        self.map_offset_x = params_map["MAP_CENTER"][0] - params_map["MAP_SIZE"][0]/2
        self.map_offset_y = params_map["MAP_CENTER"][1] - params_map["MAP_SIZE"][1]/2
        # ?????
        self.GRIDSIZE=self.map_size_x

        ## 주변 그리드를 탐색할 때 사용할 리스트
        self.dx = [-1,0,0,1,-1,-1,1,1]
        self.dy = [0,1,-1,0,-1,1,-1,1]
        self.dCost = [1,1,1,1,1.414,1.414,1.414,1.414]


    def grid_update(self):
        self.is_grid_update=True

        ## 로직 3. 맵 데이터 행렬로 바꾸기
        map_to_grid=np.array(self.map_msg.data)
        self.grid=np.reshape(map_to_grid, (self.map_size_x, self.map_size_y), order='F')


    def pose_to_grid_cell(self,x,y):

        map_point_x=0
        map_point_y=0

        ## 로직 4. 위치(x,y)를 map의 grid cell로 변환
        map_point_x= int(( x - self.map_offset_x ) / self.map_resolution)
        map_point_y= int(( y - self.map_offset_y ) / self.map_resolution)
        return map_point_x,map_point_y


    def grid_cell_to_pose(self,grid_cell):
        ## 로직 5. map의 grid cell을 위치(x,y)로 변환
        x=(grid_cell[0] * self.map_resolution) + self.map_offset_x
        y=(grid_cell[1] * self.map_resolution) + self.map_offset_y
        return [x,y]


    def odom_callback(self,msg):
        self.is_odom=True
        self.odom_msg=msg


    def map_callback(self,msg):
        self.is_map=True
        self.map_msg=msg


    def obs_callback(self, msg):
        if self.is_odom:
            print(msg.data)


    def a_star_goal_callback(self, msg):
        ## 로직 6. goal_pose 메시지 수신하여 목표 위치 설정
        goal_x=msg.x
        goal_y=msg.y

        goal_cell=self.pose_to_grid_cell(goal_x, goal_y)
        if goal_cell[0] <= self.map_size_x and goal_cell[1] <= self.map_size_y:
            self.goal = [goal_cell[0], goal_cell[1]]
        else:
            print(f'좌표가 {self.map_size_x}*{self.map_size_y}를 벗어났습니다.')


    def timer_callback(self):
        # 지도 받아왔고 odom 정보도 있는 상태에서 목적지가 정해졌다면
        if self.is_map ==True and self.is_odom==True  :

            if self.is_grid_update == False:
                self.grid_update()

            self.final_path=[]

            # 현재 위치
            x=self.odom_msg.pose.pose.position.x
            y=self.odom_msg.pose.pose.position.y
            # 터틀봇의 절대 위치를 그리드 위치로 변환하기 인 것 같다.
            start_grid_cell = self.pose_to_grid_cell(x,y)
            start_grid_cell = list(start_grid_cell)

            # 0으로 채워진 350 X 350 행렬 만들기
            self.path = [[0 for col in range(self.GRIDSIZE)] for row in range(self.GRIDSIZE)]

            # 350 * 350 으로 채워진 350 X 350 행렬 만들기
            self.cost = np.array([[self.GRIDSIZE*self.GRIDSIZE for col in range(self.GRIDSIZE)] for row in range(self.GRIDSIZE)])

            if self.grid[start_grid_cell[0]][start_grid_cell[1]] == 0 and self.grid[self.goal[0]][self.goal[1]] == 0  and start_grid_cell != self.goal :
                # 시작점을 넣어주었다.
                self.aStar(start_grid_cell)

            self.global_path_msg=Path()
            self.global_path_msg.header.frame_id='map'

            for grid_cell in reversed(self.final_path) :
                tmp_pose=PoseStamped()
                waypoint_x,waypoint_y=self.grid_cell_to_pose(grid_cell)
                tmp_pose.pose.position.x=waypoint_x
                tmp_pose.pose.position.y=waypoint_y
                tmp_pose.pose.orientation.w=1.0

                print(tmp_pose)

                self.global_path_msg.poses.append(tmp_pose)
                # print('self.global_path_msg.poses', self.global_path_msg.poses)

            # 경로가 존재한다면
            if len(self.final_path)!=0 :
                self.a_star_pub.publish(self.global_path_msg)
                print(self.final_path)


    def heuristics(self, node):
        # 스마트 홈은 경로상에서 특정한 가중치(교통 혼잡 등)을 생각할 필요가 없어서 맨하탄을 사용
        return (abs(node[0] - self.goal[0]) + abs(node[1] - self.goal[1]))


    def aStar(self,start):
        heap = []
        heapq.heappush(heap,(0,start))
        self.cost[start[0]][start[1]] = 1

        ## 로직 7. grid 기반 최단경로 탐색
        # heap 큐 시간 복잡도는 O(log n) Vs 우선순위 큐 시간 복잡도는 삽입: O(1) 탐색: O(n)
        # heap 큐를 사용
        while heap:
            cost,current = heapq.heappop(heap)
            if self.goal[0] == current[0] and self.goal[1] == current[1]:
                break

            for i in range(8):
                next = (current[0] + self.dx[i], current[1] + self.dy[i])
                if next[0] >= 0 and next[1] >= 0 and next[0] < self.GRIDSIZE and next[1] < self.GRIDSIZE:
                        if self.grid[next[0]][next[1]] <= 50:    # 장애물 확률이 50% 이하면
                            deltaCost = self.dCost[i] + self.heuristics(next)
                            if  self.cost[next[0]][next[1]] > self.cost[current[0]][current[1]] + deltaCost:
                                self.cost[next[0]][next[1]] = self.cost[current[0]][current[1]] + deltaCost
                                self.path[next[0]][next[1]] = current
                                heapq.heappush(heap, (self.cost[next[0]][next[1]], next)) # Q.append(next)

        node = self.goal
        while node != start:
            if type(node) == type(1):
                return
            nextNode = self.path[node[0]][node[1]]
            self.final_path.append(node)
            node = nextNode


def main(args=None):
    rclpy.init(args=args)

    global_planner = a_star()

    rclpy.spin(global_planner)


    global_planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


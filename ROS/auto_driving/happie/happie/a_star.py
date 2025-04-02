import rclpy
import numpy as np
from rclpy.node import Node
import os
from geometry_msgs.msg import Pose, PoseStamped, Point
from squaternion import Quaternion
from nav_msgs.msg import Odometry,OccupancyGrid,MapMetaData,Path
from math import pi,cos,sin,sqrt
import heapq
from collections import deque
from std_msgs.msg import String

from .config import params_map, PKG_PATH

# a_star 알고리즘 테스트 용
from .a_star_test import AStar

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
        super().__init__('a_Star')
        # 로직 1. publisher, subscriber 만들기
        self.map_sub = self.create_subscription(OccupancyGrid, 'map', self.map_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 1)
        self.goal_sub = self.create_subscription(PoseStamped, 'goal_pose', self.goal_callback, 1)
        self.a_star_pub = self.create_publisher(Path, 'global_path', 1)

        self.map_msg = OccupancyGrid()
        self.odom_msg = Odometry()
        self.is_map = False
        self.is_odom = False
        self.is_found_path = False
        self.is_grid_update = False

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
        self.is_grid_update = True
        # 로직 3. 맵 데이터 행렬로 바꾸기
        self.grid = np.array(self.map_msg.data).reshape(self.GRIDSIZE, self.GRIDSIZE)

    def pose_to_grid_cell(self, x, y):
        map_point_x = int((x - self.map_offset_x) / self.map_resolution)
        map_point_y = int((y - self.map_offset_y) / self.map_resolution)

        ## 로직 4. 위치(x,y)를 map의 grid cell로 변환
        map_point_x = max(0, min(self.GRIDSIZE - 1, map_point_x))
        map_point_y = max(0, min(self.GRIDSIZE - 1, map_point_y))

        return map_point_x, map_point_y

    def grid_cell_to_pose(self, grid_cell):
        ## 로직 5. map의 grid cell을 위치(x,y)로 변환
        x = grid_cell[0] * self.map_resolution + self.map_offset_x
        y = grid_cell[1] * self.map_resolution + self.map_offset_y
        return [x, y]

    def odom_callback(self, msg):
        self.is_odom = True
        self.odom_msg = msg

    def map_callback(self, msg):
        self.is_map = True
        self.map_msg = msg

    def goal_callback(self, msg):
        print('goal_pose: ', msg.pose.position.x, msg.pose.position.y)
        if msg.header.frame_id == 'map':
            goal_x = msg.pose.position.x
            goal_y = msg.pose.position.y
            goal_cell = self.pose_to_grid_cell(goal_x, goal_y)
            self.goal = goal_cell
            if self.is_map and self.is_odom:
                if not self.is_grid_update:
                    self.grid_update()
                # 시작점
                x = self.odom_msg.pose.pose.position.x
                y = self.odom_msg.pose.pose.position.y
                start_grid_cell = self.pose_to_grid_cell(x, y)

                print('Start Cell:', start_grid_cell)
                print('Goal Cell:', self.goal)

                # step 1
                print('Step 1: A* Algorithm initialization')
                astar = AStar(self.grid)
                
                # step 2
                print('Step 2: Running A* algorithm')
                path = astar.a_star(start_grid_cell, self.goal)

                # step 3
                if path:
                    print('Step 3: Path found, preparing global path message')
                    self.global_path_msg = Path()
                    self.global_path_msg.header.frame_id = 'map'

                    # step 4
                    print('Step 4: Converting path to pose')
                    for grid_cell in path:
                        tmp_pose = PoseStamped()
                        waypoint_x, waypoint_y = self.grid_cell_to_pose(grid_cell)
                        tmp_pose.pose.position.x = waypoint_x
                        tmp_pose.pose.position.y = waypoint_y
                        tmp_pose.pose.orientation.w = 1.0
                        self.global_path_msg.poses.append(tmp_pose)

                    # step 5
                    print('Step 5: Publishing the global path')
                    self.a_star_pub.publish(self.global_path_msg)

                    # step 6
                    print('Step 6: Path successfully found and published')

                # step 7
                else:
                    print('Step 7: No path found!')

                # step 8
                print('Step 8: Pathfinding process completed!')


def main(args=None):
    rclpy.init(args=args)
    global_planner = a_star()
    rclpy.spin(global_planner)
    global_planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
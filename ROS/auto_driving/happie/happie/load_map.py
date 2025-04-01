import rclpy
import numpy as np
from rclpy.node import Node

import os
from geometry_msgs.msg import Pose
from squaternion import Quaternion
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from math import pi

from .config import params_map, PKG_PATH

class loadMap(Node):

    def __init__(self):
        super().__init__('load_map')
        self.map_pub = self.create_publisher(OccupancyGrid, 'map', 1)
        
        time_period = 1  
        self.timer = self.create_timer(time_period, self.timer_callback)

        # 로직 1. 맵 파라미터 설정
        self.map_msg = OccupancyGrid()

        # Get map parameters from params_map
        self.map_size_x = int(params_map["MAP_SIZE"][0] / params_map["MAP_RESOLUTION"])
        self.map_size_y = int(params_map["MAP_SIZE"][1] / params_map["MAP_RESOLUTION"])
        self.map_resolution = params_map["MAP_RESOLUTION"]
        
        # Update offsets based on MAP_CENTER from params_map
        self.map_offset_x = params_map["MAP_CENTER"][0] - params_map["MAP_SIZE"][0]/2
        self.map_offset_y = params_map["MAP_CENTER"][1] - params_map["MAP_SIZE"][1]/2
        
        self.map_data = [0 for _ in range(self.map_size_x * self.map_size_y)]
        grid = np.array(self.map_data)
        grid = np.reshape(grid, (self.map_size_y, self.map_size_x))

        self.map_msg.header.frame_id = "map"

        # Set map metadata
        m = MapMetaData()
        m.resolution = self.map_resolution
        m.width = self.map_size_x
        m.height = self.map_size_y
        m.origin = Pose()
        m.origin.position.x = self.map_offset_x
        m.origin.position.y = self.map_offset_y
        self.map_meta_data = m
        self.map_msg.info = self.map_meta_data

        # 로직 2. 맵 데이터 읽고, 2차원 행렬로 변환
        back_folder = '..'  # 상위 폴더를 지정하려는 경우
        folder_name = 'data'  # 맵을 저장할 폴더 이름
        file_name = 'map.txt'
        full_path = os.path.join(PKG_PATH, back_folder, folder_name, file_name)

        with open(full_path, 'r') as f:
            line = f.readlines()
            line_data = line[0].split()
        
            for num, data in enumerate(line_data):
                self.map_data[num] = int(data)
   
        map_to_grid = np.array(self.map_data)
        grid = np.reshape(map_to_grid, (self.map_size_y, self.map_size_x))

        # 로직 3. 점유영역 근처 필터처리
        for y in range(self.map_size_y):
            for x in range(self.map_size_x):
                if grid[x][y] == 100:
                    for box_x in range(-5, 6):
                        for box_y in range(-5, 6):
                            if 0 < x + box_x < self.map_size_x and 0 < y + box_y < self.map_size_y and grid[x + box_x][y + box_y] < 80:
                                grid[x + box_x][y + box_y] = 127

        np_map_data = grid.reshape(1, self.map_size_x * self.map_size_y) 
        list_map_data = np_map_data.tolist()

        self.map_msg.data = list_map_data[0]

    def timer_callback(self):
        self.map_msg.header.stamp = rclpy.clock.Clock().now().to_msg()
        self.map_pub.publish(self.map_msg)

def main(args=None):
    rclpy.init(args=args)

    load_map = loadMap()
    rclpy.spin(load_map)
    load_map.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
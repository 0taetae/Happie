import rclpy
import numpy as np
import paho.mqtt.client as mqtt  # MQTT 라이브러리 추가
import json  # 데이터를 JSON으로 변환하기 위함
from rclpy.node import Node

import os
from geometry_msgs.msg import Pose
from squaternion import Quaternion
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from math import pi
from PIL import Image  # 이미지 처리 라이브러리 추가
import io
import base64  # base64로 인코딩하여 MQTT로 전송하기 위함

<<<<<<< HEAD
# from .config import params_map, PKG_PATH
params_map = {
    "MAP_RESOLUTION": 0.1,
    "OCCUPANCY_UP": 0.02,
    "OCCUPANCY_DOWN": 0.01,
    "MAP_CENTER": (0, 0),
     "MAP_SIZE": (30, 30),
    "MAP_FILENAME": 'test.png',
    "MAPVIS_RESIZE_SCALE": 2.5
}

# 맵 데이터 저장 경로 
PKG_PATH = r"C:\Users\SSAFY\Desktop\S12P21E103\ROS\auto_driving\happie\data"

=======
from .config import params_map, PKG_PATH, MQTT_CONFIG
>>>>>>> c423c0b50caf25a64561ffe1269c4d85445c2244

class loadMap(Node):

    def __init__(self):
        super().__init__('load_map')
        self.map_pub = self.create_publisher(OccupancyGrid, 'map', 1)
        
        time_period = 1  
        self.timer = self.create_timer(time_period, self.timer_callback)

        # MQTT 설정
        self.mqtt_client = mqtt.Client()
        self.mqtt_broker = MQTT_CONFIG["BROKER"]
        self.mqtt_port = MQTT_CONFIG["PORT"]
        self.mqtt_topic = "map/data"  # 퍼블리시할 토픽
        self.mqtt_client.username_pw_set(MQTT_CONFIG["USERNAME"], MQTT_CONFIG["PASSWORD"])
        # MQTT 브로커에 연결
        self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
        self.mqtt_client.loop_start()  # 비동기 처리 시작

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
        self.image_data = self.create_image_from_map(grid)

    def create_image_from_map(self, grid):
        # 맵 데이터를 이미지로 변환
        image_data = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)

        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                value = grid[y][x]  # 각 grid 값 (0-100)
                
                # 반대 그라데이션 계산: 값이 0일 때 흰색, 100일 때 검은색으로 선형 보간
                r = g = b = int((100 - value) * 2.55)  # 0에서 100 사이 값을 255에서 0 사이 값으로 변환 (100 * 2.55 = 255, 0 * 2.55 = 0)
                
                # 색상 설정
                image_data[y, x] = [r, g, b]

        # 이미지를 PIL 이미지 객체로 변환
        image = Image.fromarray(image_data)

        # 이미지를 바이트 스트림으로 변환
        image_io = io.BytesIO()
        image.save(image_io, format='PNG')  # PNG 형식으로 저장
        image_io.seek(0)  # 스트림의 시작으로 이동

        # 바이트 데이터 가져오기
        image_bytes = image_io.read()

        # base64 인코딩
        #image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # 디버깅: base64 디코딩 후 다시 이미지 변환
        #decoded_bytes = base64.b64decode(image_base64)
        #decoded_image = Image.open(io.BytesIO(decoded_bytes))

        # 디코딩된 이미지 저장 (디버깅용)
        #decoded_image_path = "map_decoded.png"
        #decoded_image.save(decoded_image_path)
        #print(f"디코딩된 이미지 저장 완료: {decoded_image_path}")

        # 이미지 직접 보기 (옵션)
        #image.show(title="원본 맵 이미지")
        #decoded_image.show(title="디코딩된 맵 이미지")

        return image_bytes

    def timer_callback(self):
        self.map_msg.header.stamp = rclpy.clock.Clock().now().to_msg()
        self.map_pub.publish(self.map_msg)

        # 이미지 데이터를 base64로 인코딩
        image_base64 = base64.b64encode(self.image_data).decode('utf-8')

        # JSON 데이터로 변환
        map_json = {
            "image": image_base64,  # base64로 인코딩된 이미지
            "params": params_map  # params_map 정보
        }

        # MQTT 퍼블리시
        self.mqtt_client.publish(self.mqtt_topic, json.dumps(map_json))
        self.get_logger().info("Map image data published via MQTT!!!!")

        # JSON 메시지 출력 (디버깅용)
        json_str = json.dumps(map_json, indent=4)
        # self.get_logger().info(f"Publishing MQTT message:\n{json_str}")

def main(args=None):
    rclpy.init(args=args)

    load_map = loadMap()
    rclpy.spin(load_map)
    load_map.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
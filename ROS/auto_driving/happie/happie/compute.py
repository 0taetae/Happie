import os
import numpy as np
import heapq
from .config import params_map, PKG_PATH


def a_star(start, goal, grid_map):
    """안전한 경로를 탐색하는 A* 알고리즘"""

    def compute_obstacle_distance_map(grid_map):
        """각 셀이 가장 가까운 장애물과의 거리를 계산하는 함수"""
        rows, cols = len(grid_map), len(grid_map[0])
        distance_map = np.full((rows, cols), np.inf)
        
        # 장애물(40 이상인 셀) 위치 저장
        obstacle_cells = [(i, j) for i in range(rows) for j in range(cols) if grid_map[i][j] >= 40]
        
        # BFS를 사용하여 각 셀과 가장 가까운 장애물과의 거리 계산
        queue = obstacle_cells[:]
        for x, y in queue:
            distance_map[x, y] = 0  # 장애물 위치는 거리 0
        
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        while queue:
            x, y = queue.pop(0)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and distance_map[nx, ny] == np.inf:
                    distance_map[nx, ny] = distance_map[x, y] + 1
                    queue.append((nx, ny))
        
        return distance_map
    
    def grid_to_real(path, params):
        grid_size = int(params["MAP_SIZE"][0] / params["MAP_RESOLUTION"])
        x_center, y_center = params["MAP_CENTER"]
        resolution = params["MAP_RESOLUTION"]
        return [
            (
                x_center + (j - grid_size // 2) * resolution,
                y_center + (i - grid_size // 2) * resolution
            )
            for i, j in path
        ]

    # 🔹 장애물 거리 맵을 미리 계산
    obstacle_distance_map = compute_obstacle_distance_map(grid_map)
    
    def get_cost(pos):
        """해당 좌표의 거리 기반 추가 비용 계산"""
        x, y = pos
        if grid_map[x][y] >= 40:
            return float('inf')  # 장애물은 절대 탐색 불가
        
        distance_to_obstacle = obstacle_distance_map[x, y]
        
        # 장애물에서 2칸 이하라면 추가 비용 (가급적 피하도록 유도)
        if distance_to_obstacle <= 2:
            return 10  
        elif distance_to_obstacle <= 4:
            return 3  
        return 1  # 일반 비용
    
    open_list = []
    closed_list = set()
    heapq.heappush(open_list, (0 + self.heuristic(start, goal), 0, self.heuristic(start, goal), start))
    
    came_from = {}
    g_score = {start: 0}
    
    while open_list:
        current_f, current_g, current_h, current_node = heapq.heappop(open_list)
        
        if current_node == goal:
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(start)
            real_path = grid_to_real(path[::-1], params_map)
            return path[::-1], real_path
        
        closed_list.add(current_node)
        
        for neighbor, base_cost in self.neighbors(current_node):
            if neighbor in closed_list:
                continue

            # 🔹 장애물 거리 기반 추가 비용 적용
            extra_cost = get_cost(neighbor)
            if extra_cost == float('inf'):  # 장애물 또는 벽이면 스킵
                continue

            tentative_g_score = current_g + base_cost + extra_cost  

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + self.heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score, tentative_g_score, self.heuristic(neighbor, goal), neighbor))
                came_from[neighbor] = current_node

    return None


pose_x, pose_y = map(int, input('현재 좌표 입력').split())
goal_x, goal_y = map(int, input('목적 좌표 입력').split())


goal_map_x = (goal_x - params_map['MAP_CENTER'][0] + params_map['MAP_SIZE'][0]/2) / params_map['MAP_RESOLUTION']
goal_map_y = (goal_y - params_map['MAP_CENTER'][1] + params_map['MAP_SIZE'][1]/2) / params_map['MAP_RESOLUTION']
# 목표 위치 확인
print(f"목표위치 맵 좌표계: map_x={goal_map_x:.0f}, map_y={goal_map_y:.0f}")

map_pose_x = (pose_x - params_map['MAP_CENTER'][0] + params_map['MAP_SIZE'][0]/2) / params_map['MAP_RESOLUTION']
map_pose_y = (pose_y - params_map['MAP_CENTER'][1] + params_map['MAP_SIZE'][1]/2) / params_map['MAP_RESOLUTION']
print(f"현 위치 맵 좌표계: map_x={goal_map_x:.0f}, map_y={goal_map_y:.0f}")

# 파일 경로 설정
back_folder = '..'  # 상위 폴더를 지정하려는 경우
#PKG_PATH = r'C:\Users\SSAFY\Desktop\S12P21E103\ROS\auto_driving\happie\happie'
folder_name = 'data'  # 맵을 저장할 폴더 이름
file_name = 'update_map.txt'  # 파일 이름
full_path = os.path.join(PKG_PATH, back_folder, folder_name, file_name)  # 전체 경로 설정

# 데이터 읽기
with open(full_path, 'r') as file:
    data = file.read().split()

# 데이터 크기 확인
grid_size = int(params_map['MAP_SIZE'][0]/params_map['MAP_RESOLUTION'])

# 1차원 배열을 NxM 크기의 2차원 배열로 변환
data_array = np.array(data, dtype=int).reshape(grid_size, grid_size)
grid = data_array
rows = len(grid)
cols = len(grid[0])
start = (int(map_pose_y), int(map_pose_x))
goal = (int(goal_map_y), int(goal_map_x))
# goal = (int(132.0), int(188.0))
path, real_path = a_star(start, goal)
print(real_path)
# 경로 표시
if path:
    # 경로를 맵에 빨간색으로 표시
    for p in path:
        data_array[p[0]][p[1]] = 50  # 경로 표시 (예: 값 50으로 표시)

# 경로가 제대로 표시되지 않으면 경로를 점으로만 표시
else:
    print("경로를 찾을 수 없습니다.")

# 만든 path를 publish
self.publish_global_path(real_path)

# 시각화 (matplotlib 사용)
fig, ax = plt.subplots()
# 먼저 전체 맵을 그립니다
cax = ax.imshow(data_array, cmap='gray', interpolation='nearest')
# 경로를 빨간색으로 그립니다
if path:
    for p in path:
        ax.plot(p[1], p[0], color='red', marker='o', markersize=2)  # 경로를 빨간색 점으로 표시

plt.colorbar(cax)  # 색상 막대 추가
plt.title("A* Pathfinding with Red Path")
plt.show()
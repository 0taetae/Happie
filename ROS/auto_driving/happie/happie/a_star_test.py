import numpy as np
import matplotlib.pyplot as plt
import heapq
import os
# from .config import params_map, PKG_PATH


params_map = {
    "MAP_RESOLUTION": 0.1,
    "OCCUPANCY_UP": 0.02,
    "OCCUPANCY_DOWN": 0.01,
    "MAP_CENTER": (-50.036964416503906, -50.065105438232),
    "MAP_SIZE": (30, 30),
    "MAP_FILENAME": 'test.png',
    "MAPVIS_RESIZE_SCALE": 2.5
}

# A* 알고리즘 클래스
class AStar:
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        
    def heuristic(self, a, b):
        # 맨해튼 거리 (거리 계산 방법을 변경할 수 있음)
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def neighbors(self, node):
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # 상, 하, 좌, 우
            (-1, -1), (-1, 1), (1, -1), (1, 1) # 대각선
        ]
        dCost = [1, 1, 1, 1, 1.414, 1.414, 1.414, 1.414] # 이동 비용 설정
        neighbors = []
        # 경계를 벗어나지 않고 벽(40 이상)이 아니면 유효한 인접 노드
        for i, direction in enumerate(directions):
            neighbor = (node[0] + direction[0], node[1] + direction[1])
            if 0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols and self.grid[neighbor[0]][neighbor[1]] < 40:
                neighbors.append((neighbor, dCost[i]))
        return neighbors

    def a_star(self, start, goal):
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
                return path[::-1]  
            
            closed_list.add(current_node)
            
            for neighbor, cost in self.neighbors(current_node):
                if neighbor in closed_list:
                    continue
                
                tentative_g_score = current_g + cost  
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score, tentative_g_score, self.heuristic(neighbor, goal), neighbor))
                    came_from[neighbor] = current_node

        return None  

# 파일 경로 설정
back_folder = '..'  # 상위 폴더를 지정하려는 경우
PKG_PATH = r'C:\Users\SSAFY\Desktop\S12P21E103\ROS\auto_driving\happie\happie'
folder_name = 'data'  # 맵을 저장할 폴더 이름
file_name = 'map.txt'  # 파일 이름
full_path = os.path.join(PKG_PATH, back_folder, folder_name, file_name)  # 전체 경로 설정

# 데이터 읽기
with open(full_path, 'r') as file:
    data = file.read().split()

# 데이터 크기 확인
grid_size = int(params_map['MAP_SIZE'][0]/params_map['MAP_RESOLUTION'])

# 1차원 배열을 NxM 크기의 2차원 배열로 변환
data_array = np.array(data, dtype=int).reshape(grid_size, grid_size)

# A* 객체 생성
astar = AStar(data_array)

# 시작점과 목표점 설정
start = (grid_size//2, grid_size//2)  # (시작 좌표)
# goal = (270, 340)     # (목표 좌표)
# goal = (281, 323)     # (목표 좌표) 281/2.5
# goal = (112, 130)
goal = (112, 140)
# goal = (151, 180)
# goal = (180, 151)
# goal = (150, 200)

# A* 알고리즘을 통해 경로 찾기
path = astar.a_star(start, goal)

# 경로 표시
if path:
    # 경로를 맵에 빨간색으로 표시
    for p in path:
        data_array[p[0]][p[1]] = 50  # 경로 표시 (예: 값 50으로 표시)

# 경로가 제대로 표시되지 않으면 경로를 점으로만 표시
else:
    print("경로를 찾을 수 없습니다.")

# 시각화 (matplotlib 사용)
fig, ax = plt.subplots()

# 먼저 전체 맵을 그립니다
cax = ax.imshow(data_array, cmap='gray', interpolation='nearest')

# 경로를 빨간색으로 그립니다
if path:
    for p in path:
        ax.plot(p[1], p[0], color='red', marker='o', markersize=2)  # 경로를 빨간색 점으로 표시

# x, y 축 눈금 설정 (10 간격)
ax.set_xticks(np.arange(0, grid_size, 10))
ax.set_yticks(np.arange(0, grid_size, 10))

# x, y 축 레이블 추가
ax.set_xticklabels(np.arange(0, grid_size, 10))
ax.set_yticklabels(np.arange(0, grid_size, 10))

plt.colorbar(cax)  # 색상 막대 추가
plt.title("A* Pathfinding with Red Path")
plt.show()



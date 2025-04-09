# 최단 경로 재계산 요청을 처리
def path_request_callback(self, msg):
    try:
        new_goal_x = msg.x
        new_goal_y = msg.y
        print(f"🔄 새로운 경로 요청: ({new_goal_x}, {new_goal_y})")

        # MQTT에서 받은 좌표를 맵 좌표계로 변환
        goal_map_x = (new_goal_x - params_map['MAP_CENTER'][0] + params_map['MAP_SIZE'][0] / 2) / params_map['MAP_RESOLUTION']
        goal_map_y = (new_goal_y - params_map['MAP_CENTER'][1] + params_map['MAP_SIZE'][1] / 2) / params_map['MAP_RESOLUTION']

        goal_map_x = int(goal_map_x)
        goal_map_y = int(goal_map_y)

        print(f"📍 변환된 목표 위치 (그리드): x={goal_map_x}, y={goal_map_y}")
        # 현재 위치를 기준으로 새로운 경로 찾기

        self.path_finding(goal_map_x, goal_map_y)

    except Exception as e:
        print(f"❌ 새로운 경로 요청 처리 오류: {e}")
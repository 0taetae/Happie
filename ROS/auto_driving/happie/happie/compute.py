# params_map = {
#     "MAP_RESOLUTION": 0.1,
#     "OCCUPANCY_UP": 0.02,
#     "OCCUPANCY_DOWN": 0.01,
#     "MAP_CENTER": (-50.036964416503906, -50.065105438232),
#     "MAP_SIZE": (30, 30),
#     "MAP_FILENAME": 'test.png',
#     "MAPVIS_RESIZE_SCALE": 2.5
# }

# def grid_to_real(path, params):
#     grid_size = int(params["MAP_SIZE"][0] / params["MAP_RESOLUTION"])  # 그리드 크기 계산
#     x_center, y_center = params["MAP_CENTER"]  # 맵 중심 좌표
#     resolution = params["MAP_RESOLUTION"]  # 해상도

#     real_path = [
#         (
#             x_center + (j - grid_size // 2) * resolution,
#             y_center + (i - grid_size // 2) * resolution
#         )
#         for i, j in path
#     ]
#     return real_path

# path = [(284, 203), (283, 203), (282, 203), (281, 204), (280, 205), (279, 206), (278, 207), (277, 208), (276, 209), (275, 210), (274, 211), (273, 212), (272, 213), (271, 214), (270, 215), (269, 216), (268, 216), (267, 216), (266, 216), (265, 216), (264, 216), (263, 216), (262, 216), (261, 216), (260, 216), (259, 216), (258, 216), (257, 216), (256, 216), (255, 216), (254, 216), (253, 217), (252, 218), (251, 217), (250, 216), (249, 216), (248, 215), (247, 214), (246, 213), (245, 212), (244, 211), (243, 210), (242, 209), (241, 208), (240, 208), (239, 207), (238, 206), (237, 205), (236, 204), (235, 203), (234, 203), (233, 202), (232, 202), (231, 202), (230, 202), (229, 202), (228, 202), (227, 202), (226, 202), (225, 202), (224, 202), (223, 202), (222, 202), (221, 202), (220, 202), (219, 201), (218, 200), (217, 199), (216, 199), (215, 199), (214, 198), (213, 198), (212, 198), (211, 198), (210, 198), (209, 198), (208, 198), (207, 198), (206, 198), (205, 198), (204, 198), (203, 198), (202, 198), (201, 198), (200, 198), (199, 198), (198, 198), (197, 198), (196, 198), (195, 198), (194, 198), (193, 198), (192, 198), (191, 198), (190, 198), (189, 198), (188, 198), (187, 198), (186, 198), (185, 198), (184, 198), (183, 198), (182, 198), (181, 198), (180, 198), (179, 198), (178, 198), (177, 198), (176, 198), (175, 198), (174, 198), (173, 198), (172, 198), (171, 198), (170, 198), (169, 198), (168, 198), (167, 198), (166, 198), (165, 198), (164, 198), (163, 197), (162, 196), (161, 195), (160, 194), (159, 193), (158, 192), (157, 191), (156, 190), (155, 189), (154, 188), (153, 187)]

# # 실제 좌표 변환
# real_path = grid_to_real(path, params_map)
# print(real_path)
# # # 변환된 좌표 출력
# # for i, (x, y) in enumerate(real_path):
# #     print(f"Point {i}: ({x:.3f}, {y:.3f})")

import math


goal_x = -49
goal_y = -50
pose_x = -50
pose_y = -50
target_heading = math.degrees(math.atan2(-(goal_x - pose_x), goal_y - pose_y))
target_heading = (target_heading + 360) % 360  # 0~360도로 변환
print('x 가 증가하는 경우', target_heading)


goal_x = -51
goal_y = -50
pose_x = -50
pose_y = -50
target_heading = math.degrees(math.atan2(-(goal_x - pose_x), goal_y - pose_y))
target_heading = (target_heading + 360) % 360  # 0~360도로 변환
print('x 가 감소하는 경우', target_heading)


goal_x = -50
goal_y = -49
pose_x = -50
pose_y = -50
target_heading = math.degrees(math.atan2(-(goal_x - pose_x), goal_y - pose_y))
target_heading = (target_heading + 360) % 360  # 0~360도로 변환
print('y 가 증가하는 경우', target_heading)


goal_x = -50
goal_y = -51
pose_x = -50
pose_y = -50
target_heading = math.degrees(math.atan2(-(goal_x - pose_x), goal_y - pose_y))
target_heading = (target_heading + 360) % 360  # 0~360도로 변환
print('y 가 감소하는 경우', target_heading)
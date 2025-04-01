import numpy as np
import cv2
import rclpy
import math
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, LaserScan

params_lidar = {
    "Range" : 90,  # min & max range of lidar azimuths
    "CHANNEL" : int(1),  # vertical channel of a lidar
    "localIP": "127.0.0.1",
    "localPort": 2368,
    "Block_SIZE": int(1206),
    "X": 0,  # meter
    "Y": 0,
    "Z": 0.4 + 0.1,
    "YAW": 0,  # deg
    "PITCH": 0,
    "ROLL": 0
}

params_cam = {
    "WIDTH": 320,  # image width
    "HEIGHT": 240,  # image height
    "FOV": 60,  # Field of view
    "localIP": "127.0.0.1",
    "localPort": 1232,
    "Block_SIZE": int(65000),
    "X": 0.,  # meter
    "Y": 0,
    "Z": 0.8,
    "YAW": 0,  # deg
    "PITCH": 0.0,
    "ROLL": 0
}

# 노드 로직 순서
# 1. 노드에 필요한 라이다와 카메라 topic의 subscriber 생성
# 2. Params를 받아서 라이다 포인트를 카메라 이미지에 projection 하는 transformation class 정의하기
# 3. 카메라 콜백함수에서 이미지를 클래스 내 변수로 저장.
# 4. 라이다 콜백함수에서 2d scan data(거리와 각도)를 가지고 x,y 좌표계로 변환
# 5. 라이다 x,y 좌표 데이터 중 정면 부분만 crop
# 6. transformation class 의 transform_lidar2cam로 라이다 포인트를 카메라 3d좌표로 변환
# 7. transformation class 의 project_pts2img 로 라이다 포인트를 2d 픽셀 좌표상으로 정사영
# 8. draw_pts_img()로 카메라 이미지에 라이다 포인트를 draw 하고 show


# Rotation matrix
def rotationMtx(yaw, pitch, roll):
    R_x = np.array([[1, 0, 0, 0],
                    [0, math.cos(roll), -math.sin(roll), 0],
                    [0, math.sin(roll), math.cos(roll), 0],
                    [0, 0, 0, 1]])
    
    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch), 0],
                    [0, 1, 0, 0],
                    [-math.sin(pitch), 0, math.cos(pitch), 0],
                    [0, 0, 0, 1]])

    R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0, 0],
                    [math.sin(yaw), math.cos(yaw), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    R = np.matmul(R_x, np.matmul(R_y, R_z))
    return R

# Translation matrix
def translationMtx(x, y, z):
    M = np.array([[1, 0, 0, x],
                  [0, 1, 0, y],
                  [0, 0, 1, z],
                  [0, 0, 0, 1]])
    return M

# Coordinate transform from lidar to camera
def transformMTX_lidar2cam(params_lidar, params_cam):
    """
    transformMTX_lidar2cam 내 좌표 변환행렬 로직 순서
    1. params에서 라이다와 카메라 센서들의 자세, 위치 정보를 뽑기.
    2. 라이다에서 카메라 위치까지 변환하는 translation 행렬을 정의
    3. 카메라의 자세로 맞춰주는 rotation 행렬을 정의.
    4. 위의 두 행렬을 가지고 최종 라이다-카메라 변환 행렬을 정의.
    """

    # 로직 1. params에서 라이다와 카메라 센서들의 자세, 위치 정보를 뽑기.
    lidar_yaw, lidar_pitch, lidar_roll = np.radians([params_lidar["YAW"], params_lidar["PITCH"], params_lidar["ROLL"]])
    cam_yaw, cam_pitch, cam_roll = np.radians([params_cam["YAW"], params_cam["PITCH"], params_cam["ROLL"]])

    lidar_pos = np.array([params_lidar["X"], params_lidar["Y"], params_lidar["Z"]])
    cam_pos = np.array([params_cam["X"], params_cam["Y"], params_cam["Z"]])

    # 로직 2. 라이다에서 카메라 까지 변환하는 translation 행렬을 정의
    Tmtx = translationMtx(lidar_pos[0] - cam_pos[0], lidar_pos[1] - cam_pos[1], lidar_pos[2] - cam_pos[2])

    # 로직 3. 카메라의 자세로 맞춰주는 rotation 행렬을 정의
    Rmtx = np.matmul(rotationMtx(cam_yaw, cam_pitch, cam_roll), rotationMtx(lidar_yaw, lidar_pitch, lidar_roll))

    # 로직 4. 위의 두 행렬을 가지고 최종 라이다-카메라 변환 행렬을 정의
    RT = np.matmul(Rmtx, Tmtx)
    return RT

    """
    테스트

    params_lidar = {
        "X": 0, # meter
        "Y": 0,
        "Z": 0.6,
        "YAW": 0, # deg
        "PITCH": 0,
        "ROLL": 0
    }


    params_cam = {
        "WIDTH": 640, # image width
        "HEIGHT": 480, # image height
        "FOV": 90, # Field of view
        "X": 0., # meter
        "Y": 0,
        "Z":  1.0,
        "YAW": 0, # deg
        "PITCH": 0.0,
        "ROLL": 0
    }

    이면

    R_T = 
    [[ 6.12323400e-17 -1.00000000e+00  0.00000000e+00  0.00000000e+00]
    [ 6.12323400e-17  3.74939946e-33 -1.00000000e+00  4.00000000e-01]
    [ 1.00000000e+00  6.12323400e-17  6.12323400e-17 -2.44929360e-17]
    [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]

    """

def project2img_mtx(params_cam):
    """
    project2img_mtx 내 projection 행렬 계산 로직 순서
    1. params에서 카메라의 width, height, fov를 가져와서 focal length를 계산.
    2. 카메라의 파라메터로 이미지 프레임 센터를 계산.
    3. Projection 행렬을 계산 

    """

    # 로직 1. params에서 카메라의 width, height, fov를 가져와서 focal length를 계산.
    fov = np.radians(params_cam["FOV"])
    fc_x = params_cam["WIDTH"] / (2 * np.tan(fov / 2))
    fc_y = params_cam["HEIGHT"] / (2 * np.tan(fov / 2))

    # 로직 2. 카메라의 파라메터로 이미지 프레임 센터를 계산.
    cx = params_cam["WIDTH"] / 2
    cy = params_cam["HEIGHT"] / 2

    # 로직 3. Projection 행렬을 계산.
    R_f = np.array([[fc_x, 0, cx],
                    [0, fc_y, cy]])
    return R_f

# Function to draw points on the image
def draw_pts_img(img, xi, yi):
    for ctr in zip(xi, yi):
        img = cv2.circle(img, ctr, 2, (255, 0, 0), -1)
    return img

class LIDAR2CAMTransform:
    def __init__(self, params_cam, params_lidar):
        # 로직 1. Params에서 필요한 파라메터들과 RT 행렬, projection 행렬 등을 정의
        self.width = params_cam["WIDTH"]
        self.height = params_cam["HEIGHT"]

        self.n = float(params_cam["WIDTH"])
        self.m = float(params_cam["HEIGHT"])

        self.RT = transformMTX_lidar2cam(params_lidar, params_cam)
        self.proj_mtx = project2img_mtx(params_cam)

    def transform_lidar2cam(self, xyz_p):
        # 로직 2. 클래스 내 self.RT로 라이다 포인트들을 카메라 좌표계로 변환시킨다.
        xyz_hom = np.hstack([xyz_p, np.ones((xyz_p.shape[0], 1))])
        xyz_c = np.dot(self.RT, xyz_hom.T).T
        return xyz_c[:, :3]

    def project_pts2img(self, xyz_c, crop=True):
        # 로직 3. RT로 좌표 변환된 포인트들의 normalizing plane 상의 위치를 계산.
        xn = xyz_c[:, 0] / xyz_c[:, 2]
        yn = xyz_c[:, 1] / xyz_c[:, 2]

        # 로직 4. normalizing plane 상의 라이다 포인트들에 proj_mtx를 곱해 픽셀 좌표값 계산.
        xyi = np.matmul(self.proj_mtx, np.vstack([xn, yn, np.ones_like(xn)]))
        xyi = xyi.T

        # 로직 5. 이미지 프레임 밖을 벗어나는 포인트들을 crop.
        if crop:
            xyi = self.crop_pts(xyi)
        return xyi

    def crop_pts(self, xyi):
        xyi = xyi[np.logical_and(xyi[:, 0] >= 0, xyi[:, 0] < self.width), :]
        xyi = xyi[np.logical_and(xyi[:, 1] >= 0, xyi[:, 1] < self.height), :]
        return xyi


class SensorCalib(Node):
    def __init__(self):
        super().__init__(node_name='ex_calib')

        # 로직 1. 노드에 필요한 라이다와 카메라 topic의 subscriber 생성
        self.subs_scan = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback, 10)

        self.subs_img = self.create_subscription(
            CompressedImage,
            '/image_jpeg/compressed',
            self.img_callback,
            10)

        # 로직 2. Params를 받아서 라이다 포인트를 카메라 이미지에 projection 하는
        # transformation class 정의하기

        self.l2c_trans = LIDAR2CAMTransform(params_cam, params_lidar)

        self.timer_period = 0.1
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.xyz, self.R, self.intens = None, None, None
        self.img = None

    def img_callback(self, msg):
        # 로직 3. 카메라 콜백함수에서 이미지를 클래스 내 변수로 저장.
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def scan_callback(self, msg):
        # 로직 4. 라이다 2d scan data(거리와 각도)를 가지고 x,y 좌표계로 변환
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        ranges = np.array(msg.ranges)

        angles = angle_min + np.arange(len(ranges)) * angle_increment
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        z = np.zeros_like(x)

        self.xyz = np.vstack([x, y, z]).T

    def timer_callback(self):
        if self.xyz is not None and self.img is not None:
            # 로직 5. 라이다 x,y 좌표 데이터 중 정면 부분만 crop
            xyz_p = self.xyz[np.abs(self.xyz[:, 1]) < 1]

            # 로직 6. transformation class 의 transform_lidar2cam 로 카메라 3d 좌표 변환
            xyz_c = self.l2c_trans.transform_lidar2cam(xyz_p)

            # 로직 7. transformation class 의 project_pts2img로 카메라 프레임으로 정사영
            xy_i = self.l2c_trans.project_pts2img(xyz_c)

            # 로직 8. draw_pts_img()로 카메라 이미지에 라이다 포인트를 draw 하고 show
            img_l2c = draw_pts_img(self.img, xy_i[:, 0].astype(int), xy_i[:, 1].astype(int))

            cv2.imshow("Lidar2Cam", img_l2c)
            cv2.waitKey(1)
        else:
            print("waiting for msg")
            pass


def main(args=None):
    rclpy.init(args=args)

    calibrator = SensorCalib()

    rclpy.spin(calibrator)


if __name__ == '__main__':
    main()

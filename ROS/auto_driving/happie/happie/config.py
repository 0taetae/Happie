# # custom map parameters
params_map = {
    "MAP_RESOLUTION": 0.1,
    "OCCUPANCY_UP": 0.02,
    "OCCUPANCY_DOWN": 0.01,
    "MAP_CENTER": (-50, -50),
    "MAP_SIZE": (30, 30),
    "MAP_FILENAME": 'test.png',
    "MAPVIS_RESIZE_SCALE": 1.0
}

# smart home 1 map parameters
# params_map = {
#     "MAP_RESOLUTION": 0.05,
#     "OCCUPANCY_UP": 0.02,
#     "OCCUPANCY_DOWN": 0.01,
#     "MAP_CENTER": (0, 0),
#     "MAP_SIZE": (17.5, 17.5),
#     "MAP_FILENAME": 'test.png',
#     "MAPVIS_RESIZE_SCALE": 2.0
# }

# 맵 데이터 저장 경로 
PKG_PATH = r"C:\Users\SSAFY\Desktop\S12P21E103\ROS\auto_driving\happie\data"

S3_BUCKET = 'ssafy-pro-bucket'  # 실제 버킷 이름
S3_FOLDER = 'fall_images'

AWS_ACCESS_KEY_ID='AKIAY2QJD24SHENZCWDY'
AWS_SECRET_ACCESS_KEY='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'


# MQTT 설정
BROKER = 'j12e103.p.ssafy.io'
PORT = 1883
TOPIC = 'fall_detection'
USERNAME = 'happie_mqtt_user'
PASSWORD = 'gkstkfckdl0411!'
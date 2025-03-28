S3_BUCKET = 'ssafy-pro-bucket'  # 실제 버킷 이름
S3_FOLDER = 'fall_images' 

s3_client = boto3.client(
    's3',
    aws_access_key_id='AKIAY2QJD24SHENZCWDY',
    aws_secret_access_key='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',
    region_name='ap-northeast-2'
)

# MQTT 설정
BROKER = "j12e103.p.ssafy.io" 
PORT = 1883
TOPIC = "fall_detection"
USERNAME = "happie_mqtt_user"
PASSWORD = "gkstkfckdl0411!"
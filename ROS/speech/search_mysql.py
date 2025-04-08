import pymysql
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# MySQL 연결 정보 (환경 변수에서 가져오기)
DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),  # 기본 포트 3306
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE"),
    "charset": "utf8mb4"
}

def get_image_for_keyword(keyword):
    """특정 키워드(5층, 간호사실 등)와 일치하는 이미지 URL 조회"""
    try:
        # MySQL 연결
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # 키워드가 포함된 행에서 'image' 컬럼 가져오기
        sql = "SELECT image FROM floor_image WHERE facility_name = %s"
        cursor.execute(sql, (keyword,))
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        # 결과가 있으면 이미지 URL 반환, 없으면 빈 문자열 반환
        return result["image"] if result else ""

    except Exception as e:
        print(f"❌ MySQL 조회 오류: {e}")
        return ""

if __name__ == "__main__":
    test_keywords = ["간호사실", "501호실", "502호실", "503호실"]

    for keyword in test_keywords:
        image_url = get_image_for_keyword(keyword)
        print(f"🔍 키워드: {keyword}, 조회된 이미지 URL: {image_url}")

### chromadb로 엑셀 파일을 벡터화 + 데이터 정제 ###

import pandas as pd
# from langchain.vectorstores import Chroma
# import chromadb
# from langchain.vectorstores import VectorStore

# 엑셀 파일 읽기
def load_excel(file_path) :
    # 엑셀 파일을 DataFrame으로 읽기
    df = pd.read_excel(r"C:\Users\SSAFY\Desktop\LLM\병원정보.xlsx")
    
    # 파일이 제대로 읽혔는지 확인
    print("엑셀 파일을 성공적으로 읽었습니다!")
        
    # 데이터의 첫 5개 행을 출력
    print("엑셀 데이터 샘플:")
    print(df.head())  # 첫 5개 행 출력
        
    # 데이터프레임의 기본 정보 출력
    print("엑셀 데이터 정보:")
    print(df.info())
    
    return df

# 함수 테스트 (엑셀 파일 경로를 입력하세요)
file_path = r"C:\Users\SSAFY\Desktop\LLM\병원정보.xlsx"  # Raw string 사용
df = load_excel(file_path)

# 파일이 정상적으로 로드되었다면 df가 DataFrame 객체입니다.
if df is not None:
    print("엑셀 파일이 정상적으로 로드되었습니다.")
else:
    print("엑셀 파일 로드 실패.")
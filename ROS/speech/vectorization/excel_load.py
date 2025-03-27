### 엑셀 파일 로드 ###

import pandas as pd

# 엑셀 파일 로드 함수
def load_excel():
    # 엑셀 파일 경로
    file_path = "C:\\Users\\SSAFY\\Desktop\\LLM\\hospital_info.xlsx"
    # 엑셀 파일을 DataFrame으로 읽기
    df = pd.read_excel(file_path, sheet_name=0)
    
    # print("엑셀 데이터 정보:")
    # print(df.info())
    
    # '위치설명' 컬럼 전처리: 쉼표로 구분된 텍스트 항목 분리
    if '위치설명' in df.columns:
        df['위치설명'] = df['위치설명'].apply(lambda x: [i.strip() for i in str(x).split(',')] if pd.notnull(x) else [])

    # 추가적인 전처리 작업 (예: 공백 제거, 결측치 처리 등)
    # 예: 공백 제거
    df['시설명'] = df['시설명'].apply(lambda x: str(x).strip() if pd.notnull(x) else "")
    df['서비스설명'] = df['서비스설명'].apply(lambda x: str(x).strip() if pd.notnull(x) else "")
    df['층정보'] = df['층정보'].apply(lambda x: str(x).strip() if pd.notnull(x) else "")
    
    return df

if __name__ == "__main__":
    df = load_excel()
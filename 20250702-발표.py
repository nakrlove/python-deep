import pandas as pd
import glob
import os
import re





def makeFile():
    merged_df = pd.concat(df_list, ignore_index=True)

    # 저장
    output_path = "./sumdata.xlsx"
    merged_df.to_excel(output_path, index=False)


# 날짜 추출 함수 정의
def extract_yyyymm(file_name):
    """
    파일명에서 6자리 연월(YYYYMM)을 추출해 'YYYY년MM월' 형식으로 반환
    예: immigration_201001.xls → '2010년01월'
    """
    match = re.search(r"(\d{4})(\d{2})", file_name)
   
    if match:
        year, month = match.group(1), match.group(2)
        return f"{year}년{month}월"
    return "날짜없음"

    
# 참고 ###############################################################
# .xls 형식의 엑셀 파일을 읽기 위해 필요한 라이브러리인 xlrd 가 설치가 필요함
# pip install xlrd 
######################################################################

# 1. 엑셀 파일들이 있는 폴더 지정
folder_path = "./sample/eData/"
pattern = "immigration_*.xls*"
comfile = ''

# 폴더 및 파일 목록
folder_path = "./sample/eData/"
excel_files = sorted(glob.glob(os.path.join(folder_path,pattern)))

if not excel_files:
    raise FileNotFoundError("엑셀 파일을 찾을 수 없습니다.")

# 결과 리스트 초기화
df_list = []

# 첫 번째 파일 처리 (헤더 추출용)
first_file = excel_files[0]
print(f"헤더 기준 파일: {first_file}")

# 첫 줄(타이틀) 건너뛰고 두 번째 줄을 헤더로 사용
df_header = pd.read_excel(first_file, header=0, skiprows=1)
# header_columns = df_header.columns.tolist()



# 첫 번째 파일 데이터 처리
period = extract_yyyymm(first_file)
df_header.insert(0, period, period)  # 첫 번째 열에 '2010년01월'이라는 이름의 컬럼 추가
# df_header[period] = os.path.basename(first_file)
df_list.append(df_header)


makeFile()


# 나머지 파일 처리
for file in excel_files[1:]:
    try:
        df = pd.read_excel(file, header=0, skiprows=2)  # 헤더 포함된 행 건너뛰기

        # if len(df.columns) != len(header_columns):
        #     print(f"{file}: 열 개수 불일치. 건너뜁니다.")
        #     continue

        # df.columns = header_columns  # 헤더 컬럼 적용

        # 해당 파일의 yyyymm 추출 후 앞 열에 추가
        period = extract_yyyymm(file)
        print(period)
        # df.insert(0, period, period)  # 맨 앞 열에 '2010년01월' 등의 컬럼과 값 추가
        # df[period] = os.path.basename(file)
        df_list.append(df)

    except Exception as e:
        print(f"{file} 처리 오류: {e}")



makeFile()
# # 데이터 통합
# merged_df = pd.concat(df_list, ignore_index=True)

# # 저장
# output_path = "./sumdata.xlsx"
# merged_df.to_excel(output_path, index=False)


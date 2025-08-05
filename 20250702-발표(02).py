import pandas as pd
import glob
import os
import re

# #####################################################################
# ########################## 1) 데이터 추출 ############################
# #####################################################################

# 2. 2020년05월(방한외래 관광객 세부통계 PART)~21년 3월까지 별도합치

folder_path = "./sample/eData/"
pattern = "*방한외래관광객 세부통계*.xls*"
# 폴더 및 파일 목록
excel_files = sorted(glob.glob(os.path.join(folder_path,pattern)))


if not excel_files:
    raise FileNotFoundError("엑셀 파일을 찾을 수 없습니다.")


# 결과 리스트 초기화
df_list = []

# 연령부분 
# 정규식: "숫자-숫자" 형식만 남기기
def is_valid_age_range(val):
    return bool(re.match(r'^\d{2}-\d{2}$', str(val)))




valid_genders = ['남성', '여성']
# 나머지 파일 처리
for file in excel_files:
  
    try:
        
        # skiprow=1 헤더 포함된 행 건너뛰기 (첫번째 레코드 skip)
        # 엑셀 sheet중 2번째 sheet를 선택해서 데이터를 추출함 sheet_name=1
        df = pd.read_excel(file,  sheet_name=1 ,engine="openpyxl")
        df.columns = ['년월','목적','연령','성별','국적','지역','입국항','교통수단','인원']
        df = df.rename(columns={"년월":"YM","목적":"purpose","연령":"age","성별":"gender","국적":"nationality","지역":"region","입국항":"airport","교통수단":"transport","인원":"count"})
        # 검색 조건 정리함  
        # - 외국인 - 
        # 주요성별
        # 연령대
        # 입국항

        # ==================================================
        # 입력된 데이터 정합성에 문제가 있음
        # 잘못된값 결측치로 처리함 
        # 1) 연령 항목 , 승무원 입력됨
        # 2) 성별 항목 , 승무원 입력됨
        # ==================================================

        # 1) 연령 항목 (승무원)
        df['age'] = df['age'].where(df['age'].apply(is_valid_age_range))
        
        # 2) 성별 항목 (승무원)
        df['gender'] = df['gender'].where(df['gender'].isin(valid_genders))

        # 결측치 제거
        df_clean = df.dropna(subset=['gender', 'age'])
        df = df.groupby(['nationality','region','airport', 'gender', 'age']).size().reset_index(name='count')
        
        print(f"{df}")
    except Exception as e:
        print(f"{file} 처리 오류: {e}")



# #####################################################################
# ########################## 2) 차트작업 ############################
# #####################################################################

import seaborn as sns
import matplotlib.pyplot as plt


def age(age_range):
    try:
        parts = age_range.split('-')
        return (int(parts[0]) + int(parts[1])) / 2
    except:
        return float('inf')  # 잘못된 값은 마지막에 정렬

df['연령중간'] = df['연령'].apply(age)
df = df.sort_values(by='연령중간')

df

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='연령', y='count', hue='성별_국적', marker='o')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.title('성별 + 국적 조합별 연령대별 입국자 수')
plt.xlabel('연령대')
plt.ylabel('입국자 수')
plt.xticks(rotation=45)
plt.legend(title='성별_국적', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
# plt.show()

sns.lineplot(data=df,x='연령',y='성별',hue='연령')
plt.show()

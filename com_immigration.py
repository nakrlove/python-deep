import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

df_1_list = []  # immigration
df_2_list = []  # 세부통계
df_3_list = []  # 목적별


plt.rcParams["font.family"] = "Malgun Gothic"  # 한글 글씨 출력
pattern = "*.xls*"


# 파일명에서 날짜부분만 추출함 
# 정규식: "숫자-숫자" 형식만 남기기
def NameSplit(val):
    match = re.search(r'(\d{6})', str(val))  # 6자리 숫자 추출
    return match.group(1) if match else None



def Info(df1 = None, df2 = None, df3 = None):
    if df1 is not  None:
        df1.info()
    if df2 is not  None:
        df2.info()
    if df3 is not  None:
        df3.info()

def dataFilter(df):
    # 데이터값 일치시킴 '유학/연수' -> '유학연수'
    df.iloc[:, 3] = df.iloc[:, 3].replace('유학/연수', '유학연수')
    return df 

# ===================================================
# 엑셀파일 목록에서 파일들 하나씩 열어 데이터를 구분해서 담기
# 1) immigration_xxxxx
# 2) xxxx 방한외래관광객 세부통계xxxx
# 3) 목적별_국적별_입국_202103_202311
# 
# ExcelList 파라메터값을 입력하지 않으면 
# Default값 "./sample/eData/"이 설정된다.
# ===================================================
# 엑셀 파일들을 분류해서 통합 처리
def ExcelList(path="./sample/eData/", pattern="*.xls*"):
  
    df_1_list.clear() 
    df_2_list.clear() 
    df_3_list.clear() 
    folder_path = path
    excel_files = sorted(glob.glob(os.path.join(folder_path, pattern)))

    if not excel_files:
        raise FileNotFoundError(f"엑셀 파일이 존재하지 않습니다: {folder_path}")

    for file in excel_files:
        file_name = os.path.basename(file)

        try:
            if "immigration" in file_name:
                temp = pd.read_excel(file, header=1)
                temp["year"] = NameSplit(file_name)
                df_1_list.append(temp)

            elif "세부통계" in file_name:
                temp = pd.read_excel(file, sheet_name=1)
                temp.rename(columns={"년월": "year"}, inplace=True)
                df_2_list.append(temp)

            elif "목적별" in file_name:
                temp = pd.read_excel(file)
                temp = temp.iloc[1:]  # 첫 줄 제거
                df_3_list.append(temp)

        except Exception as e:
            print(f"[오류] {file_name} 처리 중 에러 발생:\n {e}")

    df_1 = pd.concat(df_1_list, ignore_index=True) if df_1_list else pd.DataFrame()
    df_2 = pd.concat(df_2_list, ignore_index=True) if df_2_list else pd.DataFrame()
    df_3 = pd.concat(df_3_list, ignore_index=True) if df_3_list else pd.DataFrame()

    return df_1, df_2, df_3


# ===================================================
# 엑셀파일 목록에서 파일들 하나씩 열어 데이터를 구분해서 담기
# 
# ExcelData 파라메터값을 입력하지 않으면 
# Default값 "./sample/eData/TEST.xlsx"이 설정된다.
# ===================================================
# 엑셀 파일들을 분류해서 통합 처리
def ExcelData(path="./sample/eData/TEST2.xlsx"):
    
    df_1_list.clear()
    file_name = os.path.basename(path)
    try:
            temp = pd.read_excel(path, header=1)
            # temp["year"] = NameSplit(file_name)
            df_1_list.append(temp)

    except Exception as e:
        print(f"[오류] {file_name} 처리 중 에러 발생:\n {e}")


    # 각 리스트를 DataFrame으로 변환
    df_1 = pd.concat(df_1_list, ignore_index=True) if df_1_list else pd.DataFrame()

    return df_1


# Pie charts
# 참조
# https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_features.html#sphx-glr-gallery-pie-and-polar-charts-pie-features-py
def PieChart(df,labels):

    df = dataFilter(df)
    # 국적이 3번째 열(index 2)에 있다고 가정
    nationality_col = df.iloc[:, 2]

    # 국적 필터링
    filtered_df = df[nationality_col.isin(labels)]

    # 국적별 인원 수 집계 (인원 수 컬럼은 5번째 열이라고 가정)
    grouped = filtered_df.groupby(nationality_col)[df.columns[4]].sum()
    print(f"grouped \n {grouped} \n {grouped}")
   
    labels = grouped.index.tolist()
    sizes = grouped.values

    # 파이 차트 그리기
    fig, ax = plt.subplots()
    # ax.pie(grouped.values,  labels=labels, autopct='%1.1f%%',
    #    shadow={'ox': -0.04, 'edgecolor': 'none', 'shade': 0.9}, startangle=90)
    ax.pie(sizes,  labels=labels, autopct='%1.1f%%',
       shadow={'ox': -0.04, 'edgecolor': 'none', 'shade': 0.9}, startangle=90)
    ax.axis('equal')  # 원형 유지
    plt.show()


def LineChart(df,labels):
    # 국적 컬럼이 df의 세 번째 열(즉, index 2)라고 가정
    nationality_col = df.iloc[:, 2]

    # 국적 필터링
    filtered_df = df[nationality_col.isin(labels)]

    count_by_country = (
        filtered_df.groupby(filtered_df.iloc[:, 2])[df.columns[4]]
        .sum()
        .reindex(labels)
        .fillna(0)
        .astype(int)
    )
    
    # 메모리에 저장된 차트값들이 있다면 초기화
    plt.clf()
    plt.close()

    # 선 그래프 그리기
    plt.figure(figsize=(8,5))
    plt.plot(count_by_country.index, count_by_country.values, marker='o', linestyle='-', color='b')

    # y축 숫자값 쉼표 표시
    formatter = FuncFormatter(lambda x, _: f'{int(x):,}')
    plt.gca().yaxis.set_major_formatter(formatter)
    # 숫자에 콤마 포함해서 표시
    for i, (x, y) in enumerate(zip(count_by_country.index, count_by_country.values)):
        plt.text(x, y + 0.5, f'{y:,}', ha='center', va='bottom', fontsize=10, color='black')
    plt.title('국적별 데이터')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def PlotChart(df):
    # 미주 국가 리스트
    americas = ['미국', '캐나다', '멕시코', '브라질', '아르헨티나', '칠레', '미주 기타']

    # 날짜 컬럼 datetime으로 변환 (1번째 열)
    df.iloc[:, 1] = pd.to_datetime(df.iloc[:, 1], errors='coerce')

    # 컬럼 지정
    country_col = df.iloc[:, 2]
    count_col = df.iloc[:, 4]

    # 미주 국가만 필터링
    df_americas = df[country_col.isin(americas)]

    # 국가별 전체 인원수 합계 계산
    grouped = (
        df_americas
        .groupby(country_col)[count_col.name]
        .sum()
        .reindex(americas)  # 지정한 순서대로 정렬
        .fillna(0)
        .astype(int)
    )

    # 막대 차트 그리기
    ax = grouped.plot(kind='bar', figsize=(10, 6), color='skyblue')

    # Y축 천 단위 콤마
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))

    # 막대 위 숫자 표시
    for i, val in enumerate(grouped.values):
        ax.text(i, val + val * 0.01, f'{val:,}', ha='center', va='bottom', fontsize=10)

    ax.set_title('미주 국가별 총 방문 인원 수', fontsize=14)
    ax.set_xlabel('국가')
    ax.set_ylabel('인원 수')
    plt.tight_layout()
    plt.show()


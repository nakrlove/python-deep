import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import font_manager, rc
import os
os.getcwd()

##################################################
# 1. 데이터 불러오기 및 전처리
##################################################

# 1. CSV 파일 불러오기
try:
    df = pd.read_csv('/Users/nakrlove/Desktop/dev/python-deep/local_test/아파트(전월세)_실거래가_all.csv', encoding='utf-8', sep=';')
except UnicodeDecodeError:
    df = pd.read_csv('/Users/nakrlove/Desktop/dev/python-deep/local_test/아파트(전월세)_실거래가_all.csv', encoding='euc-kr', sep=';')

# 2. 데이터 전처리
# '계약년월'과 '계약일' 컬럼을 합쳐 '거래일' 컬럼 생성
df['거래일'] = pd.to_datetime(df['계약년월'].astype(str) + df['계약일'].astype(str).str.zfill(2), format='%Y%m%d')

# '전용면적(㎡)'을 특정 구간으로 분류 (예: 10㎡ 단위)
bins = [0, 40, 60, 85, 100, 135, 200]
labels = ['~40㎡', '40~60㎡', '60~85㎡', '85~100㎡', '100~135㎡', '135㎡~']
df['면적_구간'] = pd.cut(df['전용면적(㎡)'], bins=bins, labels=labels, right=False)

# '전월세구분' 컬럼을 '전세'와 '월세'로 구분
df_jeonse = df[df['전월세구분'] == '전세'].copy()
df_wolse = df[df['전월세구분'] == '월세'].copy()

print("데이터 전처리 완료. 각 데이터프레임의 상위 5개 행:")
print("전세 데이터:")

print(df_jeonse.head())
print("\n월세 데이터:")
print(df_wolse.head())



##################################################
# 2. 전용면적에 따른 수요량 분석 및 시각화
##################################################
# 3. 전용면적별 전/월세 수요량 분석
# 면적 구간별 거래 건수 계산
demand_jeonse = df_jeonse.groupby('면적_구간').size().reset_index(name='전세_수요량')
demand_wolse = df_wolse.groupby('면적_구간').size().reset_index(name='월세_수요량')


# 운영체제에 맞는 한글 폰트 설정
# Windows
plt.rcParams['font.family'] = 'Malgun Gothic'
# Mac OS
plt.rcParams['font.family'] = 'AppleGothic'

# 마이너스 부호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 폰트 설정 확인 (선택 사항)
# print(plt.rcParams['font.family'])

# 4. 시각화
plt.figure(figsize=(15, 6))

# 전세 수요량 막대 그래프
plt.subplot(1, 2, 1)
plt.bar(demand_jeonse['면적_구간'], demand_jeonse['전세_수요량'], color='skyblue')
plt.title('전용면적에 따른 전세 수요량')
plt.xlabel('전용면적 (㎡)')
plt.ylabel('거래량')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# 월세 수요량 막대 그래프
plt.subplot(1, 2, 2)
plt.bar(demand_wolse['면적_구간'], demand_wolse['월세_수요량'], color='salmon')
plt.title('전용면적에 따른 월세 수요량')
plt.xlabel('전용면적 (㎡)')
plt.ylabel('거래량')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



##################################################
# 3. 시계열 데이터 가공 및 10월 수요량 예측
##################################################

# 5. 시계열 데이터 생성
# 월별 거래량 집계
monthly_demand_jeonse = df_jeonse.groupby(df_jeonse['거래일'].dt.to_period('M')).size().reset_index(name='전세_수요량')
monthly_wolse_demand = df_wolse.groupby(df_wolse['거래일'].dt.to_period('M')).size().reset_index(name='월세_수요량')

monthly_demand_jeonse['거래일'] = monthly_demand_jeonse['거래일'].dt.to_timestamp()
monthly_wolse_demand['거래일'] = monthly_wolse_demand['거래일'].dt.to_timestamp()

# 6. 예측 모델 학습 (선형 회귀)
# 데이터를 숫자형으로 변환 (월 순서)
monthly_demand_jeonse['month_index'] = np.arange(len(monthly_demand_jeonse))
monthly_wolse_demand['month_index'] = np.arange(len(monthly_wolse_demand))

# 예측할 월 (2025년 10월)의 인덱스 계산
predict_date = pd.to_datetime('2025-10-01')
last_date = monthly_demand_jeonse['거래일'].max()
predict_month_index = len(monthly_demand_jeonse) + (predict_date.year - last_date.year) * 12 + (predict_date.month - last_date.month)

# 전세 수요량 예측
X_jeonse = monthly_demand_jeonse[['month_index']]
y_jeonse = monthly_demand_jeonse['전세_수요량']
model_jeonse = LinearRegression()
model_jeonse.fit(X_jeonse, y_jeonse)
predicted_jeonse_demand = model_jeonse.predict([[predict_month_index]])

# 월세 수요량 예측
X_wolse = monthly_wolse_demand[['month_index']]
y_wolse = monthly_wolse_demand['월세_수요량']
model_wolse = LinearRegression()
model_wolse.fit(X_wolse, y_wolse)
predicted_wolse_demand = model_wolse.predict([[predict_month_index]])

# 7. 결과 출력
print(f"\n2025년 10월 전세 수요량 예측: {int(predicted_jeonse_demand[0])} 건")
print(f"2025년 10월 월세 수요량 예측: {int(predicted_wolse_demand[0])} 건")

# 예측 결과를 시각화
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(monthly_demand_jeonse['거래일'], monthly_demand_jeonse['전세_수요량'], marker='o', label='실제 전세 수요량')
plt.plot(predict_date, predicted_jeonse_demand, 'r*', markersize=10, label='예측치 (2025년 10월)')
plt.title('월별 전세 수요량 및 예측')
plt.xlabel('날짜')
plt.ylabel('거래량')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(monthly_wolse_demand['거래일'], monthly_wolse_demand['월세_수요량'], marker='o', label='실제 월세 수요량')
plt.plot(predict_date, predicted_wolse_demand, 'r*', markersize=10, label='예측치 (2025년 10월)')
plt.title('월별 월세 수요량 및 예측')
plt.xlabel('날짜')
plt.ylabel('거래량')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from datetime import datetime


# 운영체제에 맞는 한글 폰트 설정
# Windows
plt.rcParams['font.family'] = 'Malgun Gothic'
# Mac OS
plt.rcParams['font.family'] = 'AppleGothic'

# 마이너스 부호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# CSV 파일 불러오기
try:
    df = pd.read_csv('/Users/nakrlove/Desktop/dev/python-deep/local_test/아파트(전월세)_실거래가_all.csv', encoding='utf-8', sep=';')
except UnicodeDecodeError:
    df = pd.read_csv('/Users/nakrlove/Desktop/dev/python-deep/local_test/아파트(전월세)_실거래가_all.csv', encoding='euc-kr', sep=';')

# 데이터 전처리 (이전과 동일)
df['거래일'] = pd.to_datetime(df['계약년월'].astype(str) + df['계약일'].astype(str).str.zfill(2), format='%Y%m%d')
df_jeonse = df[df['전월세구분'] == '전세'].copy()
df_wolse = df[df['전월세구분'] == '월세'].copy()

# 시계열 데이터 가공 (이전과 동일)
monthly_demand_jeonse = df_jeonse.groupby(df_jeonse['거래일'].dt.to_period('M')).size().reset_index(name='전세_수요량')
monthly_wolse_demand = df_wolse.groupby(df_wolse['거래일'].dt.to_period('M')).size().reset_index(name='월세_수요량')

monthly_demand_jeonse['거래일'] = monthly_demand_jeonse['거래일'].dt.to_timestamp()
monthly_wolse_demand['거래일'] = monthly_wolse_demand['거래일'].dt.to_timestamp()

# 모델 학습을 위한 데이터 준비
# '거래일'을 숫자형 변수(월 인덱스)로 변환
monthly_demand_jeonse['month_index'] = np.arange(len(monthly_demand_jeonse))
monthly_wolse_demand['month_index'] = np.arange(len(monthly_wolse_demand))

# 예측할 월 (2025년 10월)의 인덱스 계산
predict_date = pd.to_datetime('2025-10-01')
last_date = monthly_demand_jeonse['거래일'].max()
predict_month_index = len(monthly_demand_jeonse) + (predict_date.year - last_date.year) * 12 + (predict_date.month - last_date.month)

# 2. Random Forest 모델 학습 및 예측
# 전세 수요량 예측
X_jeonse = monthly_demand_jeonse[['month_index']]
y_jeonse = monthly_demand_jeonse['전세_수요량']
model_rf_jeonse = RandomForestRegressor(n_estimators=100, random_state=42) # n_estimators: 트리의 개수
model_rf_jeonse.fit(X_jeonse, y_jeonse)
predicted_rf_jeonse = model_rf_jeonse.predict([[predict_month_index]])

# 월세 수요량 예측
X_wolse = monthly_wolse_demand[['month_index']]
y_wolse = monthly_wolse_demand['월세_수요량']
model_rf_wolse = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf_wolse.fit(X_wolse, y_wolse)
predicted_rf_wolse = model_rf_wolse.predict([[predict_month_index]])

# 3. 결과 출력 및 시각화
print(f"\n랜덤 포레스트 모델 예측 결과 (2025년 10월):")
print(f"전세 수요량: {int(predicted_rf_jeonse[0])} 건")
print(f"월세 수요량: {int(predicted_rf_wolse[0])} 건")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(monthly_demand_jeonse['거래일'], monthly_demand_jeonse['전세_수요량'], marker='o', label='실제 전세 수요량')
plt.plot(predict_date, predicted_rf_jeonse, 'r*', markersize=10, label='예측치 (2025년 10월)')
plt.title('월별 전세 수요량 및 예측 (랜덤 포레스트)')
plt.xlabel('날짜')
plt.ylabel('거래량')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(monthly_wolse_demand['거래일'], monthly_wolse_demand['월세_수요량'], marker='o', label='실제 월세 수요량')
plt.plot(predict_date, predicted_rf_wolse, 'r*', markersize=10, label='예측치 (2025년 10월)')
plt.title('월별 월세 수요량 및 예측 (랜덤 포레스트)')
plt.xlabel('날짜')
plt.ylabel('거래량')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
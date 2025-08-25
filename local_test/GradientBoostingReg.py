# Gradient Boosting Regression (그레디언트 부스팅 회귀)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

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

# 데이터 전처리
df['거래일'] = pd.to_datetime(df['계약년월'].astype(str) + df['계약일'].astype(str).str.zfill(2), format='%Y%m%d')

bins = [0, 40, 60, 85, 100, 135, 200]
labels = ['~40㎡', '40~60㎡', '60~85㎡', '85~100㎡', '100~135㎡', '135㎡~']
df['면적_구간'] = pd.cut(df['전용면적(㎡)'], bins=bins, labels=labels, right=False)

df['월'] = df['거래일'].dt.to_period('M')
monthly_demand = df.groupby(['월', '전월세구분', '면적_구간']).size().reset_index(name='수요량')
monthly_demand['월'] = monthly_demand['월'].dt.to_timestamp()

# 'month_index'를 전체 데이터 기준으로 일관되게 생성
monthly_demand_full = monthly_demand.groupby('월').size().reset_index()
monthly_demand_full['month_index'] = np.arange(len(monthly_demand_full))
monthly_demand = pd.merge(monthly_demand, monthly_demand_full[['월', 'month_index']], on='월', how='left')

# 예측할 월 (2025년 10월)의 인덱스 계산
predict_date_index = (pd.to_datetime('2025-10-01').year - monthly_demand['월'].min().year) * 12 + (pd.to_datetime('2025-10-01').month - monthly_demand['월'].min().month)

print("--- 면적 구간별 Gradient Boosting Regression 예측 결과 및 성능 지표 ---")

# 예측 결과를 담을 딕셔너리
predicted_results = {'전세': {}, '월세': {}}
performance_metrics = {'전세': {}, '월세': {}}
analysis_metrics = {'전세': {}, '월세': {}}

# 전/월세 및 면적 구간별로 반복
for rent_type in ['전세', '월세']:
    print(f"\n--- {rent_type} ---")
    for area_label in labels:
        subset_df = monthly_demand[(monthly_demand['전월세구분'] == rent_type) & (monthly_demand['면적_구간'] == area_label)].copy()

        if len(subset_df) < 2:
            print(f"  > {area_label} : 데이터 부족으로 예측 불가")
            continue

        X = subset_df[['month_index']]
        y = subset_df['수요량']
        
        # GradientBoostingRegressor 모델 학습
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X, y)
        
        # 2025년 10월 예측
        predicted_demand = model.predict([[predict_date_index]])[0]
        
        # 예측 성능 평가
        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        # 특성 중요도
        feature_importance = model.feature_importances_[0]

        # 결과 저장 및 출력
        predicted_results[rent_type][area_label] = int(predicted_demand)
        performance_metrics[rent_type][area_label] = {'RMSE': rmse, 'R2': r2, 'MAE': mae}
        analysis_metrics[rent_type][area_label] = {'특성 중요도': feature_importance}
        
        print(f"  > {area_label} 예측: {int(predicted_demand)}건")
        print(f"  > {area_label} 성능 지표: RMSE={rmse:.2f}, R²={r2:.2f}, MAE={mae:.2f}")

# --- 결과 시각화 ---
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 전세 예측 결과 시각화
jeonse_demands = [predicted_results['전세'].get(label, 0) for label in labels]
axes[0].bar(labels, jeonse_demands, color='skyblue')
axes[0].set_title('2025년 10월 전세 수요량 예측 (면적 구간별)', fontsize=15)
axes[0].set_xlabel('전용면적 (㎡)', fontsize=12)
axes[0].set_ylabel('예상 거래량', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

# 월세 예측 결과 시각화
wolse_demands = [predicted_results['월세'].get(label, 0) for label in labels]
axes[1].bar(labels, wolse_demands, color='salmon')
axes[1].set_title('2025년 10월 월세 수요량 예측 (면적 구간별)', fontsize=15)
axes[1].set_xlabel('전용면적 (㎡)', fontsize=12)
axes[1].set_ylabel('예상 거래량', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
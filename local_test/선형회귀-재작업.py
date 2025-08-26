import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# 운영체제에 맞는 한글 폰트 설정
# Windows
plt.rcParams['font.family'] = 'Malgun Gothic'
# Mac OS
plt.rcParams['font.family'] = 'AppleGothic'

# 마이너스 부호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False
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
monthly_demand_full = monthly_demand.groupby('월').size().reset_index()
monthly_demand_full['month_index'] = np.arange(len(monthly_demand_full))
monthly_demand = pd.merge(monthly_demand, monthly_demand_full[['월', 'month_index']], on='월', how='left')

# 예측할 월 (2025년 10월)의 인덱스 계산
predict_date_index = (pd.to_datetime('2025-10-01').year - monthly_demand['월'].min().year) * 12 + (pd.to_datetime('2025-10-01').month - monthly_demand['월'].min().month)

print("--- 면적 구간별 Linear Regression 예측 결과 및 성능 지표 ---")

for rent_type in ['전세', '월세']:
    print(f"\n==================== {rent_type} ====================")
    for area_label in labels:
        subset_df = monthly_demand[(monthly_demand['전월세구분'] == rent_type) & (monthly_demand['면적_구간'] == area_label)].copy()

        if len(subset_df) < 6:
            print(f"  > {area_label} : 데이터 부족으로 예측 및 평가 불가")
            continue

        X = subset_df[['month_index']]
        y = subset_df['수요량']
        
        # 시계열 데이터 분할 (마지막 3개월을 테스트 데이터로 사용)
        test_size = 3
        X_train = X[:-test_size]
        print(f" X_train. == {X_train}")
        X_test = X[-test_size:]
        print(f" X_test. == {X_test}")
        y_train = y[:-test_size]
        y_test = y[-test_size:]
        
        # 모델 학습 및 예측
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        predicted_demand_future = model.predict([[predict_date_index]])[0]
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # 성능 평가
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        r2_train = r2_score(y_train, y_train_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        r2_test = r2_score(y_test, y_test_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        
        print(f"\n--- {area_label} ---")
        print(f"  [Linear Regression] 2025년 10월 예측: {int(predicted_demand_future)}건")
        print(f"  - 훈련 성능: RMSE={rmse_train:.2f}, R²={r2_train:.2f}, MAE={mae_train:.2f}")
        print(f"  - 테스트 성능: RMSE={rmse_test:.2f}, R²={r2_test:.2f}, MAE={mae_test:.2f}")
        print(f"  - 과적합 여부: {'과적합 의심' if r2_train > r2_test and r2_train - r2_test > 0.2 else '양호'}")
        print(f"  - 실제값 리스트: {list(y.astype(int))}")
        full_y_pred = np.concatenate([y_train_pred, y_test_pred])
        print(f"  - 예측값 리스트: {[int(val) for val in full_y_pred]}")

        # 예측값과 실제값 차트 시각화
        plt.figure(figsize=(10, 6))
        plt.plot(subset_df['월'], y, label='실제 수요량', color='blue', marker='o')
        plt.plot(subset_df['월'], full_y_pred, label='예측 수요량', color='red', linestyle='--')
        plt.title(f'{area_label} {rent_type} - Linear Regression 예측 및 성능')
        plt.xlabel('날짜')
        plt.ylabel('거래량')
        plt.legend()
        plt.grid(True)
        plt.show()

# 최종 2025년 10월 예측 결과 시각화
predicted_demands = {label: 0 for label in labels}
for rent_type in ['전세', '월세']:
    for area_label in labels:
        subset_df = monthly_demand[(monthly_demand['전월세구분'] == rent_type) & (monthly_demand['면적_구간'] == area_label)].copy()
        if len(subset_df) >= 6:
            X = subset_df[['month_index']]
            y = subset_df['수요량']
            model = LinearRegression()
            model.fit(X, y)
            predicted_demands[area_label] += model.predict([[predict_date_index]])[0]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('2025년 10월 Linear Regression 최종 예측 (면적 구간별)', fontsize=18)
axes[0].bar(labels, [predicted_demands[label] for label in labels], color='skyblue')
axes[0].set_title('전세 수요량', fontsize=15)
axes[0].set_xlabel('전용면적 (㎡)', fontsize=12)
axes[0].set_ylabel('예상 거래량', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
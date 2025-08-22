import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import numpy as np

# -----------------------------
# 1. 데이터 불러오기
# -----------------------------
print("데이터를 불러오는 중입니다...")
try:
    # 아파트 전월세 실거래가 데이터를 불러옵니다.
    # 사용자의 로컬 경로를 참고하여 파일 경로를 설정했습니다.
    apt_rent = pd.read_csv("C:\\Users\\Admin\\study01\\local_test\\아파트(전월세)_실거래가_all.csv", sep=';', encoding='utf-8', low_memory=False)
    
    # 자치구 단위 생활인구 데이터를 불러옵니다.
    kosis_data = pd.read_csv("C:\\Users\\Admin\\study01\\local_test\\자치구 단위 서울 생활인구(내국인).csv", sep=',', encoding='utf-8-sig')

    # 컬럼 이름의 앞뒤 공백을 제거하여 KeyError를 방지합니다.
    kosis_data.columns = kosis_data.columns.str.strip()
    
    print("데이터 불러오기 완료.")
except FileNotFoundError as e:
    print(f"파일을 찾을 수 없습니다: {e.filename}")
    exit()


# -----------------------------
# 2. 전월세 실거래가 데이터 전처리 및 거래량 집계
# -----------------------------
print("전월세 실거래가 데이터 전처리 및 거래량 집계 중...")
# '시군구'에서 자치구명 추출 (예: '서울특별시 강남구' -> '강남구')
apt_rent['자치구명'] = apt_rent['시군구'].str.extract(r'서울특별시\s*(\S+구)')

# '전용면적(㎡)' 컬럼의 데이터 타입을 숫자로 변환합니다.
apt_rent['전용면적(㎡)'] = pd.to_numeric(apt_rent['전용면적(㎡)'], errors='coerce')

# apt_rent.info()
# apt_rent.isnull().sum()

# 전용면적이 NaN 값인 행은 분석에서 제외합니다.
apt_rent.dropna(subset=['전용면적(㎡)'], inplace=True)

# '전용면적(㎡)'를 59, 84, 114 등으로 반올림하여 그룹화에 사용합니다.
apt_rent['반올림_전용면적'] = apt_rent['전용면적(㎡)'].round(0)




apt_rent = apt_rent[['전월세구분','전용면적(㎡)','계약년월','보증금(만원)','월세금(만원)','건축년도','자치구명','반올림_전용면적']]

# dd = apt_rent.groupby(['반올림_전용면적'])
# print(dd.groups.keys())
# 전세와 월세 데이터를 분리하여 각각의 거래량을 집계합니다.
apt_rent_jeonse = apt_rent[apt_rent['전월세구분'] == '전세'].copy()
apt_rent_wolse = apt_rent[apt_rent['전월세구분'] == '월세'].copy()

monthly_transactions_jeonse = apt_rent_jeonse.groupby(['자치구명', '계약년월', '반올림_전용면적']).size().reset_index(name='거래량')
monthly_transactions_wolse = apt_rent_wolse.groupby(['자치구명', '계약년월', '반올림_전용면적']).size().reset_index(name='거래량')

print("전월세 실거래가 거래량 집계 완료.")

data = apt_rent.groupby('자치구명')

# -----------------------------
# 3. 데이터 병합
# -----------------------------
print("데이터 병합 중...")
# kosis_data에서 필요한 컬럼만 선택하고 '자치구명'으로 그룹화하여 총 생활인구수를 계산합니다.
kosis_data_agg = kosis_data.groupby('자치구코드')['총생활인구수'].sum().reset_index()
kosis_data_agg.rename(columns={'총생활인구수': '인구수'}, inplace=True)

# '자치구코드'를 '자치구명'으로 매핑합니다.
gu_code_mapping = {
    11110: '종로구', 11140: '중구', 11170: '용산구', 11200: '성동구', 11215: '광진구',
    11230: '동대문구', 11260: '중랑구', 11290: '성북구', 11305: '강북구', 11320: '도봉구',
    11320: '노원구', 11380: '은평구', 11410: '서대문구', 11440: '마포구', 11470: '양천구',
    11500: '강서구', 11530: '구로구', 11545: '금천구', 11560: '영등포구', 11590: '동작구',
    11620: '관악구', 11650: '서초구', 11680: '강남구', 11710: '송파구', 11740: '강동구'
}
kosis_data_agg['자치구명'] = kosis_data_agg['자치구코드'].map(gu_code_mapping)
kosis_data_agg.set_index('자치구명', inplace=True)

# 월별 전세 거래량 데이터와 KOSIS 데이터를 '자치구명'을 기준으로 병합합니다.
merged_data_jeonse = pd.merge(monthly_transactions_jeonse, kosis_data_agg, left_on='자치구명', right_index=True, how='left')
merged_data_jeonse['인구수'].fillna(method='ffill', inplace=True)
merged_data_jeonse.dropna(inplace=True)
merged_data_jeonse['월'] = pd.to_datetime(merged_data_jeonse['계약년월'].astype(str), format='%Y%m').dt.month

# 월별 월세 거래량 데이터와 KOSIS 데이터를 '자치구명'을 기준으로 병합합니다.
merged_data_wolse = pd.merge(monthly_transactions_wolse, kosis_data_agg, left_on='자치구명', right_index=True, how='left')
merged_data_wolse['인구수'].fillna(method='ffill', inplace=True)
merged_data_wolse.dropna(inplace=True)
merged_data_wolse['월'] = pd.to_datetime(merged_data_wolse['계약년월'].astype(str), format='%Y%m').dt.month

print("데이터 병합 완료.")


# -----------------------------
# 4. 학습 데이터 준비 및 모델 학습
# -----------------------------
print("모델 학습 중...")
feature_cols = ['월', '인구수', '반올림_전용면적']

# 전세 모델 학습
X_jeonse = merged_data_jeonse[feature_cols]
y_jeonse = merged_data_jeonse['거래량']
X_train_jeonse, X_test_jeonse, y_train_jeonse, y_test_jeonse = train_test_split(X_jeonse, y_jeonse, test_size=0.2, random_state=42)
rf_model_jeonse = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf_model_jeonse.fit(X_train_jeonse, y_train_jeonse)

# 월세 모델 학습
X_wolse = merged_data_wolse[feature_cols]
y_wolse = merged_data_wolse['거래량']
X_train_wolse, X_test_wolse, y_train_wolse, y_test_wolse = train_test_split(X_wolse, y_wolse, test_size=0.2, random_state=42)
rf_model_wolse = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf_model_wolse.fit(X_train_wolse, y_train_wolse)

print("모델 학습 및 평가 완료.")
print("\n- 전세 모델 성능 -")
y_pred_jeonse = rf_model_jeonse.predict(X_test_jeonse)
rmse_jeonse = np.sqrt(mean_squared_error(y_test_jeonse, y_pred_jeonse))
r2_jeonse = r2_score(y_test_jeonse, y_pred_jeonse)
print(f"RMSE (평균 제곱근 오차): {rmse_jeonse:.2f}")
print(f"R² 점수 (결정계수): {r2_jeonse:.3f}")

print("\n- 월세 모델 성능 -")
y_pred_wolse = rf_model_wolse.predict(X_test_wolse)
rmse_wolse = np.sqrt(mean_squared_error(y_test_wolse, y_pred_wolse))
r2_wolse = r2_score(y_test_wolse, y_pred_wolse)
print(f"RMSE (평균 제곱근 오차): {rmse_wolse:.2f}")
print(f"R² 점수 (결정계수): {r2_wolse:.3f}")

# -----------------------------
# 5. 2025년 10월 전용면적별 거래량 예측
# -----------------------------
# [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0, 132.0, 133.0, 134.0, 135.0, 136.0, 137.0, 138.0, 139.0, 140.0, 141.0, 142.0, 143.0, 144.0, 145.0, 146.0, 147.0, 148.0, 149.0, 150.0, 151.0, 152.0, 153.0, 154.0, 155.0, 156.0, 157.0, 158.0, 159.0, 160.0, 161.0, 162.0, 163.0, 164.0, 165.0, 166.0, 167.0, 168.0, 169.0, 170.0, 171.0, 172.0, 173.0, 174.0, 175.0, 176.0, 177.0, 178.0, 179.0, 180.0, 181.0, 182.0, 183.0, 184.0, 185.0, 186.0, 187.0, 188.0, 189.0, 190.0, 191.0, 192.0, 193.0, 194.0, 195.0, 196.0, 197.0, 198.0, 199.0, 200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0, 209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0, 217.0, 218.0, 219.0, 220.0, 221.0, 222.0, 223.0, 224.0, 225.0, 226.0, 227.0, 228.0, 229.0, 230.0, 231.0, 232.0, 233.0, 234.0, 235.0, 236.0, 237.0, 238.0, 239.0, 240.0, 241.0, 242.0, 243.0, 244.0, 245.0, 246.0, 248.0, 250.0, 252.0, 254.0, 256.0, 257.0, 258.0, 261.0, 267.0, 273.0, 274.0, 301.0, 317.0]

print("\n2025년 10월 전용면적별 거래량 예측 중...")
target_areas = [24.0,59.0, 84.0,118.0]

# 예측 데이터 생성
idx = pd.MultiIndex.from_product([kosis_data_agg.index, target_areas], names=['자치구명', '반올림_전용면적'])
future_data = pd.DataFrame(index=idx).reset_index()
future_data['월'] = 10
future_data = pd.merge(future_data, kosis_data_agg.reset_index(), on='자치구명', how='left')

# 전세 예측
future_predictions_jeonse = rf_model_jeonse.predict(future_data[['월', '인구수', '반올림_전용면적']])
future_predictions_jeonse = np.round(future_predictions_jeonse).astype(int)
prediction_df_jeonse = future_data.copy()
prediction_df_jeonse['예측_거래량(건)'] = future_predictions_jeonse
prediction_df_jeonse_unique = prediction_df_jeonse.groupby(['자치구명', '반올림_전용면적'])['예측_거래량(건)'].mean().reset_index()
prediction_df_jeonse_unique['전월세구분'] = '전세'

# 월세 예측
future_predictions_wolse = rf_model_wolse.predict(future_data[['월', '인구수', '반올림_전용면적']])
future_predictions_wolse = np.round(future_predictions_wolse).astype(int)
prediction_df_wolse = future_data.copy()
prediction_df_wolse['예측_거래량(건)'] = future_predictions_wolse
prediction_df_wolse_unique = prediction_df_wolse.groupby(['자치구명', '반올림_전용면적'])['예측_거래량(건)'].mean().reset_index()
prediction_df_wolse_unique['전월세구분'] = '월세'

# 전세와 월세 예측 결과를 하나의 데이터프레임으로 합치기
combined_prediction_df = pd.concat([prediction_df_jeonse_unique, prediction_df_wolse_unique])

print("\n- 2025년 10월 지역별 및 전용면적별 전월세 통합 예측 거래량 -")
# 합쳐진 데이터프레임을 이용하여 피벗 테이블 생성
combined_pivot_table = pd.pivot_table(
    combined_prediction_df,
    index='자치구명',
    columns=['전월세구분', '반올림_전용면적'],
    values='예측_거래량(건)'
).fillna(0).astype(int)

print(combined_pivot_table)

# -----------------------------
# 6. 예측 결과 시각화
# -----------------------------
try:
    font_path = font_manager.findfont('Malgun Gothic')
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
except:
    print("한글 폰트 'Malgun Gothic'을 찾을 수 없어 기본 폰트로 출력합니다.")

plt.rcParams['axes.unicode_minus'] = False

# 전용면적별 전월세 통합 예측 결과 시각화
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 8))
fig.suptitle('2025년 10월 서울시 전용면적별 아파트 전월세 예측 거래량', fontsize=20)
axes = axes.flatten()

# 각 전용면적별로 전세와 월세 데이터를 하나의 차트에 그립니다.
for i, area in enumerate(target_areas):
    ax = axes[i]
    plot_df = combined_prediction_df[combined_prediction_df['반올림_전용면적'] == area]
    
    # 자치구명 기준으로 정렬하여 일관성을 유지합니다.
    jeonse_df = plot_df[plot_df['전월세구분'] == '전세'].sort_values('자치구명')
    wolse_df = plot_df[plot_df['전월세구분'] == '월세'].sort_values('자치구명')
    
    x_labels = jeonse_df['자치구명']
    x = np.arange(len(x_labels))
    width = 0.35
    
    # 전세와 월세를 나란히 그립니다.
    ax.bar(x - width/2, jeonse_df['예측_거래량(건)'], width, label='전세', color='dodgerblue')
    ax.bar(x + width/2, wolse_df['예측_거래량(건)'], width, label='월세', color='coral')

    ax.set_title(f"전용면적 {area}㎡({area*0.3025})평", fontsize=14)
    ax.set_ylabel("예측 거래량 (건)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--')
    ax.legend()
    

# tight_layout(rect=[left, bottom, right, top])
# plt.figure(figsize=(15, 7))
# plt.tight_layout(rect=[0, 2.9, 1.5, 0.96])
plt.subplots_adjust(hspace=0.4, wspace=0.3)  # hspace=세로 간격, wspace=가로 간격
# plt.subplots_adjust(left=0.15, right=0.95, bottom=0.08, top=0.9)
plt.show()

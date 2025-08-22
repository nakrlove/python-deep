import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import numpy as np
import warnings

# 경고 메시지 무시 설정
# warnings.filterwarnings('ignore')

# -----------------------------
# 1. 데이터 불러오기
# -----------------------------
print("데이터를 불러오는 중입니다...")
try:
    # 아파트 전월세 실거래가 데이터를 불러옵니다.
    apt_rent = pd.read_csv("C:\\Users\\Admin\\study01\\local_test\\아파트(전월세)_실거래가_all.csv", sep=';', encoding='utf-8', low_memory=False)
    
    # 자치구 단위 생활인구 데이터를 불러옵니다.
    kosis_data = pd.read_csv("C:\\Users\\Admin\\study01\\local_test\\자치구 단위 서울 생활인구(내국인).csv", sep=',', encoding='utf-8-sig')

    # 새로 추가할 시도별 소득 데이터를 불러옵니다.
    # 이 파일은 사용자가 직접 업로드한 파일입니다.
    # 파일명을 직접 지정하는 대신, 제공된 접근 경로를 사용합니다.
    economic_data_path = "C:\\Users\\Admin\\study01\\local_test\\시도별_1인당_지역내총생산__지역총소득__개인소득_20250822165800.csv"
    economic_data = pd.read_csv(economic_data_path, sep=',', encoding='euc-kr')

    # 컬럼 이름의 앞뒤 공백을 제거하여 KeyError를 방지합니다.
    kosis_data.columns = kosis_data.columns.str.strip()
    economic_data.columns = economic_data.columns.str.strip()
    
    print("데이터 불러오기 완료.")
except FileNotFoundError as e:
    print(f"파일을 찾을 수 없습니다: {e.filename}")
    exit()

# -----------------------------
# 2. 전월세 실거래가 데이터 전처리 및 거래금액 집계
# -----------------------------
print("전월세 실거래가 데이터 전처리 및 거래금액 집계 중...")
# '시군구'에서 자치구명 추출 (예: '서울특별시 강남구' -> '강남구')
apt_rent['자치구명'] = apt_rent['시군구'].str.extract(r'서울특별시\s*(\S+구)')

# '전용면적(㎡)' 컬럼의 데이터 타입을 숫자로 변환합니다.
apt_rent['전용면적(㎡)'] = pd.to_numeric(apt_rent['전용면적(㎡)'], errors='coerce')

# 전용면적이 NaN 값인 행은 분석에서 제외합니다.
apt_rent.dropna(subset=['전용면적(㎡)', '보증금(만원)', '월세금(만원)'], inplace=True)
# 1. 숫자로 변환 (문자 → 숫자)
apt_rent['보증금(만원)'] = pd.to_numeric(apt_rent['보증금(만원)'], errors='coerce')
apt_rent['월세금(만원)'] = pd.to_numeric(apt_rent['월세금(만원)'], errors='coerce')
# '보증금(만원)'과 '월세금(만원)'을 합하여 총 거래금액을 계산합니다.
# 월세의 경우 통상적인 전월세 전환율(예: 월세금에 12를 곱함)을 적용하여 보증금과 합산합니다.
apt_rent['총거래금액'] = apt_rent['보증금(만원)'] + apt_rent['월세금(만원)'] * 12

# '전용면적(㎡)'를 59, 84, 114 등으로 반올림하여 그룹화에 사용합니다.
apt_rent['반올림_전용면적'] = apt_rent['전용면적(㎡)'].round(0)

# 전세와 월세 데이터를 분리하여 각각의 평균 거래금액을 집계합니다.
apt_rent_jeonse = apt_rent[apt_rent['전월세구분'] == '전세'].copy()
apt_rent_wolse = apt_rent[apt_rent['전월세구분'] == '월세'].copy()

monthly_transactions_jeonse = apt_rent_jeonse.groupby(['자치구명', '계약년월', '반올림_전용면적'])['총거래금액'].mean().reset_index(name='평균_거래금액')
monthly_transactions_wolse = apt_rent_wolse.groupby(['자치구명', '계약년월', '반올림_전용면적'])['총거래금액'].mean().reset_index(name='평균_거래금액')

print("전월세 실거래가 거래금액 집계 완료.")


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
    11350: '노원구', 11380: '은평구', 11410: '서대문구', 11440: '마포구', 11470: '양천구',
    11500: '강서구', 11530: '구로구', 11545: '금천구', 11560: '영등포구', 11590: '동작구',
    11620: '관악구', 11650: '서초구', 11680: '강남구', 11710: '송파구', 11740: '강동구'
}
# 오타 수정: 노원구 코드가 11350이므로 11320에서 변경
kosis_data_agg['자치구명'] = kosis_data_agg['자치구코드'].map(gu_code_mapping)
kosis_data_agg.set_index('자치구명', inplace=True)


# 시도별 소득 데이터를 '서울특별시' 값만 추출하여 변수로 저장합니다.
seoul_economic_data = economic_data[economic_data['시도별'] == '서울특별시']
seoul_economic_features = {
    '1인당 지역내총생산': seoul_economic_data['1인당 지역내총생산'].iloc[0],
    '1인당 지역총소득': seoul_economic_data['1인당 지역총소득'].iloc[0],
    '1인당 개인소득': seoul_economic_data['1인당 개인소득'].iloc[0],
    '1인당 민간소비': seoul_economic_data['1인당 민간소비'].iloc[0]
}


# 월별 전세 거래금액 데이터와 KOSIS 데이터를 '자치구명'을 기준으로 병합합니다.
merged_data_jeonse = pd.merge(monthly_transactions_jeonse, kosis_data_agg, left_on='자치구명', right_index=True, how='left')
merged_data_jeonse['인구수'].fillna(method='ffill', inplace=True)
merged_data_jeonse.dropna(inplace=True)
merged_data_jeonse['월'] = pd.to_datetime(merged_data_jeonse['계약년월'].astype(str), format='%Y%m').dt.month

# 서울시 경제 데이터를 모든 행에 추가합니다.
for col, val in seoul_economic_features.items():
    merged_data_jeonse[col] = val

# 월별 월세 거래금액 데이터와 KOSIS 데이터를 '자치구명'을 기준으로 병합합니다.
merged_data_wolse = pd.merge(monthly_transactions_wolse, kosis_data_agg, left_on='자치구명', right_index=True, how='left')
merged_data_wolse['인구수'].fillna(method='ffill', inplace=True)
merged_data_wolse.dropna(inplace=True)
merged_data_wolse['월'] = pd.to_datetime(merged_data_wolse['계약년월'].astype(str), format='%Y%m').dt.month

# 서울시 경제 데이터를 모든 행에 추가합니다.
for col, val in seoul_economic_features.items():
    merged_data_wolse[col] = val

print("데이터 병합 완료.")


# -----------------------------
# 4. 학습 데이터 준비 및 모델 학습
# -----------------------------
print("모델 학습 중...")
# 추가된 경제 데이터를 포함하여 특성 컬럼을 업데이트합니다.
feature_cols = ['월', '인구수', '반올림_전용면적'] + list(seoul_economic_features.keys())

# 전세 모델 학습
X_jeonse = merged_data_jeonse[feature_cols]
y_jeonse = merged_data_jeonse['평균_거래금액']
X_train_jeonse, X_test_jeonse, y_train_jeonse, y_test_jeonse = train_test_split(X_jeonse, y_jeonse, test_size=0.2, random_state=42)
rf_model_jeonse = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf_model_jeonse.fit(X_train_jeonse, y_train_jeonse)

# 월세 모델 학습
X_wolse = merged_data_wolse[feature_cols]
y_wolse = merged_data_wolse['평균_거래금액']
merged_data_wolse.info()
X_train_wolse, X_test_wolse, y_train_wolse, y_test_wolse = train_test_split(X_wolse, y_wolse, test_size=0.2, random_state=42)
rf_model_wolse = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf_model_wolse.fit(X_train_wolse, y_wolse)

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
# 5. 2025년 8, 9, 10월 전용면적별 거래금액 예측
# -----------------------------
print("\n2025년 8, 9, 10월 전용면적별 거래금액 예측 중...")
target_areas = [59.0, 84.0, 114.0, 130.0]
future_months = [8, 9, 10]

# 예측 데이터 생성
idx = pd.MultiIndex.from_product([kosis_data_agg.index, target_areas, future_months], names=['자치구명', '반올림_전용면적', '월'])
future_data = pd.DataFrame(index=idx).reset_index()
future_data = pd.merge(future_data, kosis_data_agg.reset_index(), on='자치구명', how='left')

# 예측 데이터에 서울시 경제 데이터를 추가합니다.
for col, val in seoul_economic_features.items():
    future_data[col] = val

# 전세 예측
future_predictions_jeonse = rf_model_jeonse.predict(future_data[feature_cols])
future_predictions_jeonse = np.round(future_predictions_jeonse).astype(int)
prediction_df_jeonse = future_data.copy()
prediction_df_jeonse['예측_거래금액(만원)'] = future_predictions_jeonse
prediction_df_jeonse_unique = prediction_df_jeonse.groupby(['자치구명', '반올림_전용면적'])['예측_거래금액(만원)'].mean().reset_index()
prediction_df_jeonse_unique['전월세구분'] = '전세'

# 월세 예측
future_predictions_wolse = rf_model_wolse.predict(future_data[feature_cols])
future_predictions_wolse = np.round(future_predictions_wolse).astype(int)
prediction_df_wolse = future_data.copy()
prediction_df_wolse['예측_거래금액(만원)'] = future_predictions_wolse
prediction_df_wolse_unique = prediction_df_wolse.groupby(['자치구명', '반올림_전용면적'])['예측_거래금액(만원)'].mean().reset_index()
prediction_df_wolse_unique['전월세구분'] = '월세'

# 전세와 월세 예측 결과를 하나의 데이터프레임으로 합치기
combined_prediction_df = pd.concat([prediction_df_jeonse_unique, prediction_df_wolse_unique])

print("\n- 2025년 8~10월 지역별 및 전용면적별 전월세 통합 예측 거래금액 -")
# 합쳐진 데이터프레임을 이용하여 피벗 테이블 생성
combined_pivot_table = pd.pivot_table(
    combined_prediction_df,
    index='자치구명',
    columns=['전월세구분', '반올림_전용면적'],
    values='예측_거래금액(만원)'
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
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
fig.suptitle('2025년 8~10월 서울시 전용면적별 아파트 전월세 예측 평균 거래금액', fontsize=20)
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
    ax.bar(x - width/2, jeonse_df['예측_거래금액(만원)'], width, label='전세', color='dodgerblue')
    ax.bar(x + width/2, wolse_df['예측_거래금액(만원)'], width, label='월세', color='coral')

    ax.set_title(f"전용면적 {area}㎡", fontsize=15)
    ax.set_ylabel("예측 평균 거래금액 (만원)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--')
    ax.legend()
    
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


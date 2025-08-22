import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder


#########################################################################
# 아파트 관리비 예측
# CSV 파일로 예측해보기
#########################################################################

# 가상 데이터 생성
# data = {
#     '전용면적(㎡)': [84, 59, 114, 84, 59, 114, 84, 59, 114],
#     '건축년도': [2000, 2010, 2015, 2000, 2010, 2015, 2000, 2010, 2015],
#     '층': [5, 12, 20, 10, 15, 25, 3, 8, 18],
#     '난방방식': ['개별', '개별', '중앙', '중앙', '개별', '중앙', '개별', '개별', '중앙'],
#     '관리비(만원)': [20, 15, 35, 25, 18, 40, 19, 17, 38]
# }

data = {
    '전용면적(㎡)': [59],
    '건축년도': [ 2010],
    '층': [ 15 ],
    '난방방식': [ '개별'],
    '관리비(만원)': [ 15]
}

df = pd.DataFrame(data)

# 범주형 변수(난방방식) 원핫 인코딩
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(df[['난방방식']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['난방방식']))
df = pd.concat([df.drop('난방방식', axis=1), encoded_df], axis=1)

# 특성(X)과 타겟(y) 분리
X = df.drop('관리비(만원)', axis=1)
y = df['관리비(만원)']

# 학습 및 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("### 문제 1: 아파트 관리비 예측\n")
print(f"Mean Squared Error: {mse:.2f}")


"""
#############################################################
###################부동산 매물별 적정 호가 예측######################
#############################################################
"""
#import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 데이터 불러오기
file_path = 'C:\\Users\\Admin\\study01\\20250819\\아파트(매매)_실거래가_20250819144622-ver0.2.csv'
df = pd.read_csv(file_path, encoding='utf-8', sep=';')

# 필요한 컬럼만 선택
df = df[['시군구', '전용면적(㎡)', '계약년월', '층', '건축년도', '거래금액(만원)']]

# 데이터 전처리 및 결측치 제거
df = df.dropna()
df['거래금액(만원)'] = df['거래금액(만원)'].str.replace(',', '').astype(int)

# 피처 엔지니어링
df['계약년'] = df['계약년월'].astype(str).str[:4].astype(int)
df['계약월'] = df['계약년월'].astype(str).str[4:].astype(int)
df['경과년수'] = df['계약년'] - df['건축년도']

# 범주형 변수 인코딩
le_sigungu = LabelEncoder()
df['시군구_인코딩'] = le_sigungu.fit_transform(df['시군구'])

# 특성과 타겟 변수 분리
features = ['전용면적(㎡)', '층', '경과년수', '시군구_인코딩', '계약년', '계약월']
X = df[features]
y = df['거래금액(만원)']

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# XGBoost 모델 정의 및 학습
model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 예측 및 성능 평가
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("### 부동산 매물별 적정 호가 예측 (XGBoost)\n")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} 만원")
print(f"R^2 Score: {r2:.4f}")




#####################################################################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
import numpy as np

# 데이터 불러오기
file_path = 'C:\\Users\\Admin\\study01\\20250819\\아파트(매매)_실거래가_20250819144622-ver0.2.csv'
df = pd.read_csv(file_path, encoding='utf-8', sep=';')


# 필요한 컬럼만 선택
df = df[['시군구', '전용면적(㎡)', '계약년월', '층', '건축년도', '거래금액(만원)']]

# 데이터 전처리 및 결측치 제거
df = df.dropna()
df['거래금액(만원)'] = df['거래금액(만원)'].str.replace(',', '').astype(int)

# --- ★★★ 수정된 로직: '시군구' 컬럼 정돈 ★★★ ---
df['시군구'] = df['시군구'].apply(lambda x: ' '.join(x.split()[:2]))
# 예시: '서울특별시 강남구 개포동' -> '서울특별시 강남구' 로 변경
# ----------------------------------------------------

# 피처 엔지니어링
df['계약년'] = df['계약년월'].astype(str).str[:4].astype(int)
df['계약월'] = df['계약년월'].astype(str).str[4:].astype(int)
df['경과년수'] = df['계약년'] - df['건축년도']

# 범주형 변수 인코딩
le_sigungu = LabelEncoder()
df['시군구_인코딩'] = le_sigungu.fit_transform(df['시군구'])

# 특성과 타겟 변수 분리
features = ['전용면적(㎡)', '층', '경과년수', '시군구_인코딩', '계약년', '계약월']
X = df[features]
y = df['거래금액(만원)']

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# XGBoost 모델 정의 및 학습
model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
model.fit(X_scaled, y)

# --- 새로운 매물에 대한 예측 ---

print("### 부동산 매물 적정 호가 예측기\n")

# 예측하고자 하는 매물의 정보 입력
input_sigungu = "서울특별시 강남구" 
input_area = 84.9
input_year_built = 2018
input_floor = 15

# 현재 년월 기준으로 경과년수와 계약년월 계산
current_year = 2025
current_month = 8
input_age = current_year - input_year_built

# 입력 값 전처리
try:
    input_sigungu_encoded = le_sigungu.transform([input_sigungu])[0]
except ValueError:
    print(f"오류: '{input_sigungu}'는 학습 데이터에 없는 지역입니다. 예측할 수 없습니다.")
    exit()

# 예측을 위한 데이터프레임 생성
input_data = pd.DataFrame([[input_area, input_floor, input_age, input_sigungu_encoded, current_year, current_month]], columns=features)

# 입력 데이터 스케일링
input_scaled = scaler.transform(input_data)

# 적정 호가 예측
predicted_price = model.predict(input_scaled)[0]

# 결과 출력
print(f"**예측 매물 정보**")
print(f"  - 지역: {input_sigungu}")
print(f"  - 전용면적: {input_area}㎡")
print(f"  - 건축년도: {input_year_built}년")
print(f"  - 경과년수: {input_age}년")
print(f"  - 층: {input_floor}층")
print(f"\n**예측 적정 호가: {predicted_price:,.0f}만원**")

# 예측을 위한 데이터프레임 생성 (학습 데이터와 동일한 형식)
input_data = pd.DataFrame([[input_area, input_floor, input_age, input_sigungu_encoded, current_year, current_month]], columns=features)

# 입력 데이터 스케일링
input_scaled = scaler.transform(input_data)

# 적정 호가 예측
predicted_price = model.predict(input_scaled)[0]

# 결과 출력
print(f"**예측 매물 정보**")
print(f"  - 지역: {input_sigungu}")
print(f"  - 전용면적: {input_area}㎡")
print(f"  - 건축년도: {input_year_built}년")
print(f"  - 경과년수: {input_age}년")
print(f"  - 층: {input_floor}층")
print(f"\n**예측 적정 호가: {predicted_price:,.0f}만원**")
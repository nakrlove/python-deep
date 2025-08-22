import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

import os



# 아파트 매매 
os.getcwd()
# 데이터 불러오기
file_path = 'C:\\Users\\Admin\\study01\\20250819\\아파트(매매)_실거래가_20250819144622-ver0.2.csv'
df = pd.read_csv(file_path, encoding='euc-kr', sep=';')

# 필요한 컬럼만 선택
df = df[['시군구', '단지명', '전용면적(㎡)', '계약년월', '층', '건축년도', '거래금액(만원)']]

# 결측치 제거
df = df.dropna()
# 숫자표현이지만 문자형 숫자여서 ,가 삽입된 부분은 에러가 발생함
df['거래금액(만원)'] = df['거래금액(만원)'].str.replace(',', '').astype(int)


# 데이터 타입 변환
df['계약년월'] = pd.to_datetime(df['계약년월'], format='%Y%m')

# 피처 엔지니어링
df['계약년'] = df['계약년월'].dt.year
df['계약월'] = df['계약년월'].dt.month
df['경과년수'] = df['계약년'] - df['건축년도']

# 불필요한 컬럼 제거
df = df.drop(columns=['계약년월', '건축년도'])

# 범주형 변수 인코딩
categorical_features = ['시군구', '단지명']
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 특성과 타겟 변수 분리
X = df.drop('거래금액(만원)', axis=1)
y = df['거래금액(만원)']

# 데이터 정규화 (스케일링)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 모델 정의 및 학습
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 성능 평가
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


# 평가 지표:
# Mean Squared Error (MSE): 예측값과 실제값 사이의 오차 제곱 평균입니다. 값이 작을수록 모델의 예측 정확도가 높음을 의미합니다.
# Root Mean Squared Error (RMSE): MSE에 제곱근을 취한 값으로, 실제 오차와 같은 단위를 가져 해석이 용이합니다.
# R² Score: 모델이 종속 변수 분산을 얼마나 잘 설명하는지 나타냅니다. 0에서 1 사이의 값이며, 1에 가까울수록 모델이 데이터를 잘 설명합니다.

print(f'Mean Squared Error: {mse:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')
print(f'R^2 Score: {r2:.2f}')
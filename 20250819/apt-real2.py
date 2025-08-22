import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


# 아파트 실거래
# 데이터 불러오기
file_path = 'C:\\Users\\Admin\\study01\\20250819\\아파트(전월세)_실거래가_20250819144737-ver0.2.csv'
df = pd.read_csv(file_path, encoding='euc-kr', sep=';')

# 필요한 컬럼만 선택
df = df[['시군구', '단지명', '전용면적(㎡)', '계약년월', '층', '건축년도', '거래금액(만원)']]

# 결측치 제거
df = df.dropna()

# '거래금액(만원)' 컬럼의 콤마(,) 제거 및 숫자형으로 변환
df['거래금액(만원)'] = df['거래금액(만원)'].str.replace(',', '').astype(int)

# 데이터 타입 변환
df['계약년월'] = pd.to_datetime(df['계약년월'], format='%Y%m')

# 피처 엔지니어링
df['계약년'] = df['계약년월'].dt.year
df['계약월'] = df['계약년월'].dt.month
df['경과년수'] = df['계약년'] - df['건축년도']
df['층'] = df['층'].astype(int)

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
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, max_depth=20)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 성능 평가
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')
print(f'R^2 Score: {r2:.2f}')
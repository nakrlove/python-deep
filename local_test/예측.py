import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
# file_path = '아파트(매매)_실거래가_20250819144622-ver0.2.csv'
file_path = 'C:\\Users\\Admin\\study01\\20250819\\아파트(전월세)_실거래가_20250819144737-ver0.2.csv'
df = pd.read_csv(file_path, encoding='utf-8', sep=';')

# 필요한 컬럼만 선택
df = df[['시군구', '전용면적(㎡)', '계약년월', '층', '건축년도', '거래금액(만원)']]

# 데이터 전처리 및 결측치 제거
df = df.dropna()
df['거래금액(만원)'] = df['거래금액(만원)'].str.replace(',', '').astype(int)

# '시군구' 컬럼 정돈
df['시군구'] = df['시군구'].apply(lambda x: ' '.join(x.split()[:2]))

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

# 예측
y_pred = model.predict(X_test)

# --- ★★★ 예측 결과를 차트로 시각화 ★★★ ---

plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.3}, line_kws={'color':'red', 'lw':2})
plt.title('예측 가격 vs. 실제 가격', fontsize=16)
plt.xlabel('실제 가격 (만원)', fontsize=12)
plt.ylabel('예측 가격 (만원)', fontsize=12)
plt.grid(True)
plt.show()

# 잔차(오차) 분포 시각화
plt.figure(figsize=(8, 5))
sns.histplot(y_test - y_pred, kde=True, bins=50)
plt.title('잔차 분포 (예측 오차)', fontsize=16)
plt.xlabel('오차 (만원)', fontsize=12)
plt.ylabel('빈도', fontsize=12)
plt.grid(True)
plt.show()
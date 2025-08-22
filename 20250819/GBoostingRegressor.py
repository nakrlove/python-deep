import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


############################################################################################
# 구별 전월세 금액대별 수요량 및 예측
############################################################################################
# 데이터 불러오기
file_path = 'C:\\Users\\Admin\\study01\\20250819\\아파트(전월세)_실거래가_20250819144737-ver0.2.csv'
df_rent = pd.read_csv(file_path, encoding='euc-kr', sep=';')

# '보증금(만원)' 컬럼 전처리
df_rent['보증금(만원)'] = pd.to_numeric(df_rent['보증금(만원)'].str.replace(',', ''), errors='coerce')
df_rent = df_rent.dropna(subset=['보증금(만원)'])

# 보증금 구간화 (1억 단위)
bins = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, np.inf]
labels = ['1억미만', '1-2억', '2-3억', '3-4억', '4-5억', '5-6억', '6-7억', '7-8억', '8-9억', '9-10억', '10억이상']
df_rent['금액대'] = pd.cut(df_rent['보증금(만원)'], bins=bins, labels=labels, right=False)

# 시군구와 금액대별 거래량 집계
df_demand = df_rent.groupby(['시군구', '금액대'] ,observed=True).size().reset_index(name='수요량')

# 인코딩
le_gu = LabelEncoder()
le_price = LabelEncoder()
df_demand['시군구_인코딩'] = le_gu.fit_transform(df_demand['시군구'])
df_demand['금액대_인코딩'] = le_price.fit_transform(df_demand['금액대'])

# 모델 학습
X = df_demand[['시군구_인코딩', '금액대_인코딩']]
y = df_demand['수요량']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_demand = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model_demand.fit(X_train, y_train)

# 예측 (예시: 특정 구의 특정 금액대 수요 예측)
sample_gu = '강남구'
sample_price = '10억이상'
sample_gu_encoded = le_gu.transform([f'서울특별시 {sample_gu}'])[0]
sample_price_encoded = le_price.transform([sample_price])[0]
predicted_demand = model_demand.predict([[sample_gu_encoded, sample_price_encoded]])[0]

print("\n### 구별 전월세 금액대별 수요량 예측 (GradientBoostingRegressor)\n")
print(f'예측 대상: {sample_gu}의 {sample_price} 금액대')
print(f'예측 수요량: {predicted_demand:.2f} 건')
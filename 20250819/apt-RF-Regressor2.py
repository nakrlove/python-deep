#############################################################
#전용면적에 따른 전월세 수요 및 예측
#############################################################
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 데이터 불러오기
file_path = 'C:\\Users\\Admin\\study01\\20250819\\아파트(전월세)_실거래가_20250819144737-ver0.2.csv'
df_rent = pd.read_csv(file_path, encoding='utf-8', sep=';')

# '전용면적(㎡)' 컬럼 전처리
df_rent['전용면적(㎡)'] = pd.to_numeric(df_rent['전용면적(㎡)'], errors='coerce')
df_rent = df_rent.dropna(subset=['전용면적(㎡)'])

# 면적 구간화 (소형: 60㎡ 이하, 중형: 60㎡~85㎡, 대형: 85㎡ 초과)
bins_area = [0, 60, 85, np.inf]
labels_area = ['소형', '중형', '대형']
df_rent['면적대'] = pd.cut(df_rent['전용면적(㎡)'], bins=bins_area, labels=labels_area, right=False)

# 면적대별 수요량 집계
df_area_demand = df_rent.groupby('면적대',observed=True).size().reset_index(name='수요량')

# 인코딩
le_area = LabelEncoder()
df_area_demand['면적대_인코딩'] = le_area.fit_transform(df_area_demand['면적대'])

# 모델 학습
X = df_area_demand[['면적대_인코딩']]
y = df_area_demand['수요량']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_area = RandomForestRegressor(n_estimators=100, random_state=42)
model_area.fit(X, y)

# 예측 (예시: 소형 면적대 수요 예측)
sample_area = '소형'
sample_area_encoded = le_area.transform([sample_area])[0]
predicted_demand = model_area.predict([[sample_area_encoded]])[0]

print("\n### 전용면적에 따른 전월세 수요 및 예측 (RandomForestRegressor)\n")
print(f'예측 대상: {sample_area} 면적대')
print(f'예측 수요량: {predicted_demand:.2f} 건')
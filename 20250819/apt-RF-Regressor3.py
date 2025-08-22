import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

###############################################################
#구별 전월세 거래량 추이 및 예측
###############################################################
# 데이터 불러오기
file_path = 'C:\\Users\\Admin\\study01\\20250819\\아파트(전월세)_실거래가_20250819144737-ver0.2.csv'
df_rent = pd.read_csv(file_path, encoding='utf-8', sep=';')

# 필요한 컬럼 선택 및 전처리
df_rent['시군구'] = df_rent['시군구'].apply(lambda x: x.split()[1])
df_rent['계약년월'] = pd.to_datetime(df_rent['계약년월'], format='%Y%m')

# 구별 월별 거래량 집계
df_gu_monthly = df_rent.groupby(['시군구', df_rent['계약년월'].dt.to_period('M')]).size().reset_index(name='거래량')
df_gu_monthly['계약년월'] = df_gu_monthly['계약년월'].dt.to_timestamp()

# 각 구별 예측 모델 학습
gu_list = df_gu_monthly['시군구'].unique()
gu_models = {}

print("### 구별 전월세 거래량 예측 (RandomForestRegressor)\n")
for gu in gu_list:
    gu_df = df_gu_monthly[df_gu_monthly['시군구'] == gu].copy()
    gu_df['월_순서'] = (gu_df['계약년월'].dt.year - gu_df['계약년월'].dt.year.min()) * 12 + gu_df['계약년월'].dt.month
    
    X = gu_df[['월_순서']]
    y = gu_df['거래량']
    
    if len(X) > 1:
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        gu_models[gu] = model
        
        # 최신 데이터를 기반으로 다음 달 예측
        last_month = X['월_순서'].max()
        next_month_pred = model.predict([[last_month + 1]])[0]
        print(f'{gu} 다음 달 예측 거래량: {next_month_pred:.2f} 건')
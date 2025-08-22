##############################################################
# 공휴일 전후 부동산 제품 수요 및 예측
##############################################################
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import date
# from holidays_kr import get_holidays_in_year
import holidays
# 데이터 불러오기
file_path = 'C:\\Users\\Admin\\study01\\20250819\\아파트(전월세)_실거래가_20250819144737-ver0.2.csv'
df_rent = pd.read_csv(file_path, encoding='utf-8', sep=';')

# 날짜 컬럼 전처리
df_rent['계약일자'] = pd.to_datetime(df_rent['계약년월'].astype(str) + df_rent['계약일'].astype(str).str.zfill(2), format='%Y%m%d', errors='coerce')
df_rent = df_rent.dropna(subset=['계약일자'])

# 공휴일 정보 추가
# holidays = get_holidays_in_year(2020) + get_holidays_in_year(2021) + get_holidays_in_year(2022) + get_holidays_in_year(2023) + get_holidays_in_year(2024) + get_holidays_in_year(2025)
# holiday_dates = set(dt.strftime('%Y-%m-%d') for dt in holidays)

# 2020~2025년 한국 공휴일 가져오기
kr_holidays = holidays.KR(years=range(2020, 2026))
# 날짜를 'YYYY-MM-DD' 형식의 문자열로 변환하여 set 생성
holiday_dates = set(dt.strftime('%Y-%m-%d') for dt in kr_holidays.keys())


df_rent['공휴일여부'] = df_rent['계약일자'].dt.strftime('%Y-%m-%d').isin(holiday_dates).astype(int)
df_rent['주말여부'] = df_rent['계약일자'].dt.dayofweek.isin([5, 6]).astype(int)
df_rent['요일'] = df_rent['계약일자'].dt.dayofweek

# 수요량 집계
df_daily_demand = df_rent.groupby(['계약일자']).size().reset_index(name='수요량')
df_daily_demand['요일'] = df_daily_demand['계약일자'].dt.dayofweek
df_daily_demand['공휴일여부'] = df_daily_demand['계약일자'].dt.strftime('%Y-%m-%d').isin(holiday_dates).astype(int)

# 특성과 타겟 변수 분리
X = df_daily_demand[['요일', '공휴일여부']]
y = df_daily_demand['수요량']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model_holiday = RandomForestRegressor(n_estimators=100, random_state=42)
model_holiday.fit(X_train, y_train)

# 예측 (예시: 특정 요일의 공휴일 수요 예측)
# 0:월, 1:화, ..., 4:금, 5:토, 6:일
sample_day = 4  # 금요일
is_holiday = 1  # 공휴일

predicted_demand = model_holiday.predict([[sample_day, is_holiday]])[0]

print("\n### 공휴일 전후 부동산 제품 수요 및 예측 (RandomForestRegressor)\n")
print(f'예측 대상: {is_holiday} 공휴일의 {sample_day} 요일 수요')
print(f'예측 수요량: {predicted_demand:.2f} 건')
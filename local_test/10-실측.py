import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder
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
    # '공동주택_아파트_정보' 파일을 제외하고 전월세 실거래가 데이터만 불러옵니다.
    # 파일 인코딩을 'cp949'로 유지하고 오류가 발생할 경우 문자를 대체하도록 설정합니다.
    apt_rent = pd.read_csv("C:\\Users\\Admin\\study01\\local_test\\아파트(전월세)_실거래가_all.csv", sep=';', encoding='cp949', errors='replace', low_memory=False)
    
    # 첨부된 통계청 데이터를 불러옵니다.
    # 파일명과 구분자가 일치하는지 확인해 주세요.
    # 인코딩을 'utf-8-sig'로 변경하여 헤더 파싱 오류를 해결합니다.
    kosis_data = pd.read_csv("C:\\Users\\Admin\\study01\\local_test\\자치구 단위 서울 생활인구(내국인).csv", sep=',', encoding='utf-8-sig')

    # 컬럼 이름의 앞뒤 공백을 제거하여 'KeyError'를 방지합니다.
    kosis_data.columns = kosis_data.columns.str.strip()
    
    # 행정동코드의 앞 5자리를 추출하여 자치구코드로 사용합니다.
    kosis_data['자치구코드'] = kosis_data['행정동코드'].astype(str).str[:5].astype(int)

    # 자치구 코드와 이름을 매핑하는 딕셔너리를 만듭니다.
    gu_code_mapping = {
        11010: '종로구', 11020: '중구', 11030: '용산구', 11040: '성동구', 11050: '광진구',
        11060: '동대문구', 11070: '중랑구', 11080: '성북구', 11090: '강북구', 11100: '도봉구',
        11110: '노원구', 11120: '은평구', 11130: '서대문구', 11140: '마포구', 11150: '양천구',
        11160: '강서구', 11170: '구로구', 11180: '금천구', 11190: '영등포구', 11200: '동작구',
        11210: '관악구', 11220: '서초구', 11230: '강남구', 11240: '송파구', 11250: '강동구'
    }
    kosis_data['자치구명'] = kosis_data['자치구코드'].map(gu_code_mapping)

    # 동일한 자치구 내의 모든 행정동 인구수를 합산합니다.
    kosis_data = kosis_data.groupby('자치구명')['총생활인구수'].sum().reset_index()
    kosis_data.rename(columns={'총생활인구수': '인구수'}, inplace=True)
    kosis_data.set_index('자치구명', inplace=True)

    print("데이터 불러오기 완료.")
except FileNotFoundError as e:
    print(f"파일을 찾을 수 없습니다: {e.filename}")
    exit()

# -----------------------------
# 2. 전월세 실거래가 데이터 전처리 및 거래량 집계
# -----------------------------
print("전월세 실거래가 데이터 전처리 및 거래량 집계 중...")
# '시군구'와 '계약년월' 컬럼을 이용하여 월별 거래량을 계산합니다.
apt_rent_sel = apt_rent[['시군구', '계약년월']].copy()
apt_rent_sel['계약년월'] = pd.to_datetime(apt_rent_sel['계약년월'].astype(str), format='%Y%m')

# '시군구'와 '계약년월' 기준으로 그룹화하여 거래 건수를 집계합니다.
monthly_transactions = apt_rent_sel.groupby(['시군구', '계약년월']).size().reset_index(name='거래량')

print("전월세 실거래가 거래량 집계 완료.")


# -----------------------------
# 3. 데이터 병합
# -----------------------------
print("데이터 병합 중...")
# 월별 거래량 데이터와 KOSIS 데이터를 '시군구'와 '자치구명'을 기준으로 병합합니다.
merged_data = pd.merge(monthly_transactions, kosis_data, left_on='시군구', right_index=True, how='left')

# KOSIS 데이터가 없는 경우를 대비해 결측치를 처리합니다.
# 실제 데이터가 있다면 이 부분은 필요 없을 수 있습니다.
merged_data.fillna(method='ffill', inplace=True)
merged_data.dropna(inplace=True)

# 시계열 모델을 위한 파생 변수 생성
merged_data['월'] = merged_data['계약년월'].dt.month
merged_data['년'] = merged_data['계약년월'].dt.year

print("데이터 병합 완료.")


# -----------------------------
# 4. 학습 데이터 준비 및 모델 학습
# -----------------------------
print("모델 학습 중...")
# 예측에 사용할 특성(Features)과 목표(Target) 변수 정의
# 첨부된 파일에는 '1인가구비율'과 '평균소득'이 없으므로 '인구수'만 사용합니다.
feature_cols = ['월', '인구수']

# 거래량을 예측 목표로 설정합니다.
X = merged_data[feature_cols]
y = merged_data['거래량']

# 학습용과 테스트용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤포레스트 모델 초기화 및 학습
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 모델 성능 평가
y_pred = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("모델 학습 및 평가 완료.")
print("\n- 모델 성능 -")
print(f"RMSE (평균 제곱근 오차): {rmse:.2f}")
print(f"R² 점수 (결정계수): {r2:.3f}")

# -----------------------------
# 5. 2025년 10월 거래량 예측
# -----------------------------
print("\n2025년 10월 거래량 예측 중...")
# 2025년 10월의 예측을 위한 데이터프레임을 생성합니다.
future_data = pd.DataFrame({
    '월': [10] * len(kosis_data),
    '인구수': kosis_data['인구수'].values,
}, index=kosis_data.index)

# 예측 수행
future_predictions = rf_model.predict(future_data)
future_predictions = np.round(future_predictions).astype(int)

# 예측 결과를 데이터프레임으로 정리
prediction_df = pd.DataFrame({'자치구명': future_data.index, '예측_거래량(건)': future_predictions})
print("\n- 2025년 10월 지역별 예측 거래량 -")
print(prediction_df)

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

# 시각화를 위한 데이터 준비
plot_data = merged_data.groupby('계약년월')['거래량'].sum().reset_index()
plot_data['계약년월'] = plot_data['계약년월'].dt.to_period('M')

# 예측 데이터 추가
future_transactions = prediction_df['예측_거래량(건)'].sum()
future_date = pd.Period('2025-10', freq='M')
future_row = pd.DataFrame([{'계약년월': future_date, '거래량': future_transactions}])
plot_data = pd.concat([plot_data, future_row], ignore_index=True)

# 시각화
plt.figure(figsize=(15, 7))
plt.plot(plot_data['계약년월'].astype(str), plot_data['거래량'], marker='o', linestyle='-', label='월별 총 거래량')
plt.scatter(future_date.strftime('%Y-%m'), future_transactions, color='red', s=100, zorder=5, label='2025년 10월 예측값')
plt.title("서울시 월별 전월세 총 거래량 및 2025년 10월 예측", fontsize=16)
plt.xlabel("계약년월", fontsize=12)
plt.ylabel("거래량 (건)", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

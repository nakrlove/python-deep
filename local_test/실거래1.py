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
# 사용자 PC 경로 대신 환경 내 파일 이름으로 불러오기
apt_info = pd.read_csv("C:\\Users\\Admin\\study01\\local_test\\서울시_공동주택_아파트_정보(단지정보).csv", sep=';', encoding='utf-8')
apt_rent = pd.read_csv("C:\\Users\\Admin\\study01\\local_test\\아파트(전월세)_실거래가_all.csv", sep=';', encoding='utf-8', low_memory=False)

# -----------------------------
# 2. 단지정보 컬럼명 정리 및 전처리
# -----------------------------
print("단지정보 데이터 전처리 중...")
# 주소 컬럼 통합
apt_info["주소"] = (
    apt_info["시도"].astype(str) + " " +
    apt_info["시군구"].astype(str) + " " +
    apt_info["읍면동"].astype(str) + " " +
    apt_info["나머지주소"].astype(str)
).str.strip()

# 컬럼명 통일을 위한 매핑 딕셔너리 생성
rename_info = {
    "아파트명": "단지명",
    "주소": "법정동",
    "전체세대수": "세대수",
    "주차대수": "주차대수", # 주차 컬럼이 없어 주차대수 사용
    "사용검사일-사용승인일": "건축년도"
}

# 컬럼명 변경
apt_info = apt_info.rename(columns=rename_info)

# 필요한 컬럼만 선택하고 결측치 제거
apt_info_sel = apt_info[['단지명','법정동','세대수','주차대수','건축년도']].dropna()
print("단지정보 데이터 전처리 완료.")


# -----------------------------
# 3. 실거래가(전월세) 컬럼명 정리 및 전처리
# -----------------------------
print("전월세 실거래가 데이터 전처리 중...")
# 전월세 데이터의 컬럼명 통일을 위한 매핑 딕셔너리 생성
# '거래금액' 대신 '보증금'을 타겟으로 설정합니다.
rename_rent = {
    "단지명": "단지명",
    "전용면적(㎡)": "전용면적(㎡)",
    "층": "층",
    "계약년월": "계약년월",
    "보증금(만원)": "보증금(만원)", # 보증금을 타겟으로 사용
    "월세금(만원)": "월세금(만원)", # 월세도 피처로 활용 가능
    "건축년도": "건축년도" # 실거래가 파일에도 건축년도가 존재하여 사용
}

apt_rent = apt_rent.rename(columns=rename_rent)
apt_rent_sel = apt_rent[['단지명', '전용면적(㎡)', '층', '보증금(만원)', '월세금(만원)', '계약년월', '건축년도', '전월세구분']].copy()

# 데이터 타입 정리
apt_rent_sel['보증금(만원)'] = apt_rent_sel['보증금(만원)'].astype(str).str.replace(',', '').astype(float)
apt_rent_sel['월세금(만원)'] = apt_rent_sel['월세금(만원)'].astype(str).str.replace(',', '').astype(float)

# 월세 데이터 중 보증금이 1000만원 미만인 행은 제외합니다.
# 이는 월세 계약에서 보증금이 큰 의미가 없는 경우를 걸러내기 위함입니다.
apt_rent_sel = apt_rent_sel[(apt_rent_sel['전월세구분'] == '전세') | (apt_rent_sel['보증금(만원)'] >= 1000)]
print("전월세 실거래가 데이터 전처리 완료.")

# -----------------------------
# 4. 데이터 병합
# -----------------------------
print("데이터 병합 중...")
data = pd.merge(apt_rent_sel, apt_info_sel, on="단지명", how="left")

# 결측치 확인 및 제거 (단지정보와 병합 후)
data = data.dropna(subset=["건축년도_x", "세대수", "주차대수"])

# 건축년도, 계약년월의 데이터 타입 정리
data["건축년도"] = data["건축년도_x"].astype(int) # 실거래가 파일의 건축년도 사용
data["계약년도"] = data["계약년월"] // 100
data["계약월"] = data["계약년월"] % 100
print("데이터 병합 완료.")

# -----------------------------
# 5. 파생 변수 (Feature Engineering)
# -----------------------------
print("파생 변수 생성 중...")
# 건축연차: 계약 시점까지의 건물 나이
data['건축연차'] = data['계약년도'] - data['건축년도']
# 층수 비율: 전체 층수 30을 기준으로 한 상대적 위치
data['평균층'] = data['층'] / 30
# 전용면적대: 면적을 10㎡ 단위로 그룹화
data['전용면적대'] = data['전용면적(㎡)'] // 10

# '법정동' 컬럼을 병합된 데이터프레임에서 가져옵니다.
data = data.dropna(subset=['법정동'])
print("파생 변수 생성 완료.")

# -----------------------------
# 6. 범주형 변수 인코딩
# -----------------------------
print("범주형 변수 인코딩 중...")
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = encoder.fit_transform(data[['법정동']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['법정동']))
data = pd.concat([data.reset_index(drop=True), encoded_df], axis=1)
print("범주형 변수 인코딩 완료.")


# -----------------------------
# 7. 학습 데이터 준비 (로그 변환)
# -----------------------------
print("학습 데이터 준비 중...")
feature_cols = [
    '전용면적(㎡)', '층', '건축연차', '평균층', '세대수', '주차대수', '전용면적대', '월세금(만원)'
] + list(encoded_df.columns)

# NaN 값이 있는 행 제거
data = data.dropna(subset=feature_cols)

# 특성 데이터
X = data[feature_cols]
# 타겟 데이터: 보증금(만원). 데이터의 편차를 줄이기 위해 로그 변환을 적용합니다.
y = np.log1p(data['보증금(만원)'])

# 학습용과 테스트용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("학습 데이터 준비 완료.")

# -----------------------------
# 8. 랜덤포레스트 학습
# -----------------------------
print("모델 학습 중...")
# 최적의 성능을 위해 n_estimators와 max_depth를 설정
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_log = rf_model.predict(X_test)
print("모델 학습 완료.")

# -----------------------------
# 9. 모델 평가
# -----------------------------
# 로그 변환된 예측값과 실제값을 원래 단위로 복원합니다.
y_pred = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)

# RMSE와 R² 점수 계산
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
r2 = r2_score(y_test_orig, y_pred)
print(f"\n랜덤포레스트 모델 성능:")
print(f"RMSE (제곱근 평균 제곱 오차): {rmse:.2f} 만원")
print(f"R² 점수 (결정계수): {r2:.3f}")

# 교차검증 (5-Fold)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X, y, cv=kf, scoring='r2')
print(f"5-Fold CV R² 평균: {cv_scores.mean():.3f}")


# -----------------------------
# 10. Feature Importance 시각화
# -----------------------------
# 한글 폰트 설정
try:
    font_path = font_manager.findfont('Malgun Gothic')
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
except:
    print("한글 폰트 'Malgun Gothic'을 찾을 수 없습니다. 기본 폰트로 출력합니다.")
    
# 마이너스(-) 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 특성 중요도 추출 및 정렬
importances = rf_model.feature_importances_
feat_imp = pd.DataFrame({'feature': X.columns, 'importance': importances})
feat_imp = feat_imp.sort_values('importance', ascending=False).reset_index(drop=True)

plt.figure(figsize=(12, 8))
plt.barh(feat_imp['feature'][:20], feat_imp['importance'][:20])
plt.gca().invert_yaxis()
plt.title("특성 중요도 (상위 20개)")
plt.xlabel("중요도")
plt.ylabel("특성")
plt.show()

# -----------------------------
# 11. 실제 vs 예측 시각화
# -----------------------------
plt.figure(figsize=(8, 8))
plt.scatter(y_test_orig, y_pred, alpha=0.5)
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2)
plt.xlabel("실제 보증금(만원)")
plt.ylabel("예측 보증금(만원)")
plt.title("실제 vs 예측 보증금")
plt.grid(True)
plt.show()

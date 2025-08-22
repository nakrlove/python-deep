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
    # 환경 내 파일 이름으로 CSV 파일 불러오기
    apt_info = pd.read_csv("서울시_공동주택_아파트_정보(단지정보).csv", sep=';', encoding='utf-8')
    apt_rent = pd.read_csv("아파트(전월세)_실거래가_all.csv", sep=';', encoding='utf-8', low_memory=False)
    print("데이터 불러오기 완료.")
except FileNotFoundError as e:
    print(f"파일을 찾을 수 없습니다: {e.filename}")
    exit()

# -----------------------------
# 2. 단지정보 데이터 전처리
# -----------------------------
print("단지정보 데이터 전처리 중...")
# '시도', '시군구', '읍면동', '나머지주소' 컬럼을 결합하여 '법정동' 컬럼 생성
apt_info["법정동"] = (
    apt_info["시도"].astype(str) + " " +
    apt_info["시군구"].astype(str) + " " +
    apt_info["읍면동"].astype(str) + " " +
    apt_info["나머지주소"].astype(str)
).str.strip()

# 컬럼명을 통일성 있게 변경
apt_info.rename(columns={
    "아파트명": "단지명",
    "전체세대수": "세대수",
    "주차대수": "주차대수",
    "사용검사일-사용승인일": "건축년도"
}, inplace=True)

# 필요한 컬럼만 선택하고 결측치가 있는 행 제거
apt_info_sel = apt_info[['단지명', '법정동', '세대수', '주차대수', '건축년도']].dropna()
print("단지정보 데이터 전처리 완료.")


# -----------------------------
# 3. 전월세 실거래가 데이터 전처리
# -----------------------------
print("전월세 실거래가 데이터 전처리 중...")
# 필요한 컬럼만 선택
apt_rent_sel = apt_rent[['단지명', '전용면적(㎡)', '층', '보증금(만원)', '월세금(만원)', '계약년월', '전월세구분']].copy()

# '보증금(만원)'과 '월세금(만원)' 컬럼의 데이터 타입 정리
apt_rent_sel['보증금(만원)'] = apt_rent_sel['보증금(만원)'].astype(str).str.replace(',', '').astype(float)
apt_rent_sel['월세금(만원)'] = apt_rent_sel['월세금(만원)'].astype(str).str.replace(',', '').astype(float)

# '월세금'이 0인 경우를 전세로 간주하고, '보증금'이 0인 경우를 월세로 간주합니다.
# 예측의 정확도를 높이기 위해 '전월세구분' 컬럼을 기준으로 데이터를 정리합니다.
apt_rent_sel = apt_rent_sel.dropna(subset=['단지명', '전용면적(㎡)', '층', '보증금(만원)', '월세금(만원)', '계약년월'])

print("전월세 실거래가 데이터 전처리 완료.")


# -----------------------------
# 4. 데이터 병합 및 파생 변수 생성
# -----------------------------
print("데이터 병합 및 파생 변수 생성 중...")
# 두 데이터프레임을 '단지명'을 기준으로 병합
data = pd.merge(apt_rent_sel, apt_info_sel, on="단지명", how="left")

# 병합 후 단지정보 데이터가 없는 행 제거
data = data.dropna(subset=["세대수", "주차대수", "건축년도"])

# '건축년도' 및 '계약년월' 데이터 타입 정리 및 파생 변수 생성
data["건축년도"] = pd.to_datetime(data["건축년도"], errors="coerce").dt.year.astype(int)
data["계약년도"] = data["계약년월"] // 100
data["건축연차"] = data["계약년도"] - data["건축년도"]

# 데이터의 특성을 반영하는 새로운 파생 변수 생성
data['평균층'] = data['층'] / 30
data['전용면적대'] = data['전용면적(㎡)'] // 10
print("데이터 병합 및 파생 변수 생성 완료.")


# -----------------------------
# 5. 학습 데이터 준비
# -----------------------------
print("학습 데이터 준비 중...")
# 범주형 변수인 '법정동'을 원-핫 인코딩
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = encoder.fit_transform(data[['법정동']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['법정동']))

# 인코딩된 데이터와 기존 데이터 병합
data = pd.concat([data.reset_index(drop=True), encoded_df], axis=1)

# 모델에 사용할 특성(Features)과 목표(Target) 변수 정의
# '월세금(만원)'을 보증금 예측의 중요한 피처로 추가
feature_cols = [
    '전용면적(㎡)', '층', '건축연차', '평균층', '세대수', '주차대수', '전용면적대', '월세금(만원)'
] + list(encoded_df.columns)

X = data[feature_cols]
# '보증금(만원)'을 예측 목표로 설정. 로그 변환으로 데이터 편차 완화
y = np.log1p(data['보증금(만원)'])

# 학습용과 테스트용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("학습 데이터 준비 완료.")


# -----------------------------
# 6. 랜덤포레스트 모델 학습 및 평가
# -----------------------------
print("모델 학습 중...")
# 랜덤포레스트 모델 초기화 및 학습
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 예측 수행 및 로그 변환 복원
y_pred_log = rf_model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)

# 모델 성능 평가 지표 계산
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
r2 = r2_score(y_test_orig, y_pred)
print("모델 학습 완료.")
print("\n- 모델 성능 -")
print(f"RMSE (평균 제곱근 오차): {rmse:.2f} 만원")
print(f"R² 점수 (결정계수): {r2:.3f}")

# 교차 검증 (Cross-validation)으로 모델 안정성 평가
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X, y, cv=kf, scoring='r2')
print(f"5-Fold 교차 검증 R² 평균: {cv_scores.mean():.3f}")


# -----------------------------
# 7. 특성 중요도 시각화
# -----------------------------
try:
    font_path = font_manager.findfont('Malgun Gothic')
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
except:
    print("한글 폰트 'Malgun Gothic'을 찾을 수 없어 기본 폰트로 출력합니다.")
    
plt.rcParams['axes.unicode_minus'] = False

importances = rf_model.feature_importances_
feat_imp = pd.DataFrame({'feature': X.columns, 'importance': importances})
feat_imp = feat_imp.sort_values('importance', ascending=False).reset_index(drop=True)

plt.figure(figsize=(12, 8))
plt.barh(feat_imp['feature'][:20], feat_imp['importance'][:20])
plt.gca().invert_yaxis()
plt.title("특성 중요도 (상위 20개)", fontsize=16)
plt.xlabel("중요도", fontsize=12)
plt.ylabel("특성", fontsize=12)
plt.show()

# -----------------------------
# 8. 실제 vs 예측 보증금 시각화
# -----------------------------
plt.figure(figsize=(8, 8))
plt.scatter(y_test_orig, y_pred, alpha=0.5, color='b', label='예측 데이터')
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2, label='이상적인 예측선')
plt.xlabel("실제 보증금(만원)", fontsize=12)
plt.ylabel("예측 보증금(만원)", fontsize=12)
plt.title("실제 보증금 vs 예측 보증금", fontsize=16)
plt.grid(True)
plt.legend()
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import numpy as np

# -----------------------------
# 1. 데이터 불러오기
# -----------------------------


# 단지정보
apt_info = pd.read_csv("C:\\Users\\Admin\\study01\\local_test\\서울시_공동주택_아파트_정보(단지정보).csv", sep=';' ,encoding='utf-8')

# 매매 실거래가
apt_price = pd.read_csv("C:\\Users\\Admin\\study01\\local_test\\아파트(매매)_실거래가_all.csv", sep=';', encoding='utf-8')

# 전월세 실거래가 (참고용, 이번 예제에서는 매매가 기준)
apt_rent = pd.read_csv("C:\\Users\\Admin\\study01\\local_test\\아파트(전월세)_실거래가_all.csv", sep=';', encoding='utf-8',low_memory=False)


print("단지정보 컬럼:", apt_info.columns.tolist())
print("실거래가 컬럼:", apt_price.columns.tolist())

# -----------------------------
# 2. 단지정보 컬럼명 정리
# (실제 파일 컬럼명을 찾아 자동 매핑)
# -----------------------------


apt_info["주소"] = (
    apt_info["시도"].astype(str) + " " +
    apt_info["시군구"].astype(str) + " " +
    apt_info["읍면동"].astype(str) + " " +
    apt_info["나머지주소"].astype(str)
).str.strip()


rename_info = {}
for col in apt_info.columns:
    if "아파트명" == col :
        rename_info[col] = "단지명"
    elif "주소" == col:
        rename_info[col] = "법정동"
    elif "전체세대수" == col:
        rename_info[col] = "세대수"
    elif "주차" == col:
        rename_info[col] = "주차대수"

    elif "사용검사일-사용승인일" == col :
    # elif "사용승인" in col or "건축" in col:
        rename_info[col] = "건축년도"

apt_info = apt_info.rename(columns=rename_info)
apt_info.info()
# -----------------------------
# 3. 실거래가 컬럼명 정리
# -----------------------------
rename_price = {}
for col in apt_price.columns:
    if "아파트" in col or "단지" in col:
        rename_price[col] = "단지명"
    elif "전용" in col:
        rename_price[col] = "전용면적(㎡)"
    elif "층" == col:
        rename_price[col] = "층"
    elif "거래금액" in col:
        rename_price[col] = "거래금액(만원)"
    elif "년" in col and "월" in col:  # 계약년월
        rename_price[col] = "계약년월"

apt_price = apt_price.rename(columns=rename_price)
apt_price.info()
# -----------------------------
# 4. 필요한 컬럼만 선택
# -----------------------------
apt_info_sel = apt_info[['단지명','법정동','세대수','주차대수','건축년도']].dropna()
apt_price_sel = apt_price[['단지명','전용면적(㎡)','층','거래금액(만원)','계약년월']].dropna()

apt_price_sel.info()
apt_price_sel['거래금액(만원)'] = apt_price_sel['거래금액(만원)'].str.replace(',', '').astype(float)
# -----------------------------
# 5. 데이터 병합 및 파생변수
# -----------------------------
apt_price_sel['거래년도'] = apt_price_sel['계약년월'] // 100
apt_price_sel['거래월'] = apt_price_sel['계약년월'] % 100



# 컬럼이 중복중
apt_info.info()
print(apt_info.columns[apt_info.columns.duplicated()])

# 병합
data = pd.merge(apt_price_sel, apt_info_sel, on="단지명", how="left")

data.info()
data.iloc[:,[-1]]



# 1. 건축년도 → datetime 변환
data["건축년도"] = pd.to_datetime(data["건축년도"], errors="coerce")

# 2. 연도(int)만 추출
data["건축년도"] = data["건축년도"].dt.year

# 3. 결측치 제거 (건축년도 없는 행 drop)
data = data.dropna(subset=["건축년도"])

# 4. int 타입으로 변환
data["건축년도"] = data["건축년도"].astype(int)

# 파생 변수
data['건축연차'] = data['거래년도'] - data['건축년도']
data['평균층'] = data['층'] / 30
data['전용면적대'] = data['전용면적(㎡)'] // 10

# -----------------------------
# 6. 범주형 변수 인코딩 (법정동)
# -----------------------------
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = encoder.fit_transform(data[['법정동']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['법정동']))




# 병합
data = pd.concat([data.reset_index(drop=True), encoded_df], axis=1)
data.columns
# -----------------------------
# 7. 학습 데이터 준비
# -----------------------------
feature_cols = [
    '전용면적(㎡)','층','건축연차','평균층','세대수','주차대수','전용면적대'
] + list(encoded_df.columns)

X = data[feature_cols]
y = data['거래금액(만원)']


data.info()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 8. 랜덤포레스트 학습
# -----------------------------
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"✅ 랜덤포레스트 RMSE: {rmse:.2f} 만원")

# -----------------------------
# 9. 피처 중요도 시각화
# -----------------------------
importances = rf_model.feature_importances_
feat_imp = pd.DataFrame({'feature': X.columns, 'importance': importances})
feat_imp = feat_imp.sort_values('importance', ascending=False)

plt.figure(figsize=(10,6))
plt.barh(feat_imp['feature'][:15], feat_imp['importance'][:15])
plt.gca().invert_yaxis()
plt.title("Feature Importance (Top 15)")
plt.show()

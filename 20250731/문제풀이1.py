
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
df = sns.load_dataset('penguins').dropna()

# 2. 표준화할 대상 선택
X = df[['body_mass_g']]
# 3. 표준화 객체 생성
scaler = StandardScaler()

# 4. 표준화 적용
X_scaled = scaler.fit_transform(X)

# 5. 결과를 DataFrame으로 보기 좋게 변환
df['body_mass_g_scaled'] = X_scaled

# 6. 비교 출력
print(df[['body_mass_g', 'body_mass_g_scaled']].head())

# 7. 시각화 (원래 값 vs 표준화 값)
plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
sns.histplot(df['body_mass_g'], kde=True)
plt.title('원래 body_mass_g 분포')

plt.subplot(1, 2, 2)
sns.histplot(df['body_mass_g_scaled'], kde=True, color='orange')
plt.title('표준화된 body_mass_g 분포')

plt.tight_layout()
plt.show()


std = df['body_mass_g'].std()
print("표준편차:", round(std, 2))

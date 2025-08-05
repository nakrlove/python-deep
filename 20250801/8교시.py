import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 데이터 로딩
url = "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/refs/heads/master/winequality-red.csv"
df = pd.read_csv(url, delimiter=";")
df = pd.read_csv(url, sep=";")


print("결측치 수:", df['quality'].isnull().sum())  # 결측치 확인

# 특성과 타겟 분리
#X = df.drop('quality', axis=1)
X = df.iloc[:, :11]
X.info()
y = df['quality']

# 스케일링 (KNN, 로지스틱 회귀 등에서 성능 향상 가능)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 훈련/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 여러 분류기 테스트
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SGD": SGDClassifier(max_iter=1000, tol=1e-3),
    "Decision Tree": DecisionTreeClassifier(criterion='entropy', max_depth=5),
}

best_model = None
best_acc = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} 정확도: {acc*100:.2f}%")
    
    if acc > best_acc:
        best_acc = acc
        best_model = name

print(f"\n 최고 정확도 모델: {best_model} ({best_acc*100:.2f}%)")
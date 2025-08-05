

import pandas as pd



# 아래 wine - quality 데이터셋에 대하여, 최고 정확도를 달성해 봅시다.
url = "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/refs/heads/master/winequality-red.csv"
#df = pd.read_csv(url,delimiter=";")
df = pd.read_csv(url,sep=";")
df.info()



X = df.drop('quality', axis=1)
y = df['quality']
df['quality'].isnull().sum()
df = df.dropna()

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier #확률적 경사하강법
from sklearn.tree import DecisionTreeClassifier#결정 트리

from sklearn.metrics import accuracy_score

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test,y_train, y_test = train_test_split( X,y, test_size=0.2)

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










kn = KNeighborsClassifier(n_neighbors=3)
#kn = DecisionTreeClassifier(criterion='entropy',max_depth=2,min_samples_split=5,min_samples_leaf=2)
kn.fit(X_train,y_train)
preds = kn.predict(X_test)
acc = accuracy_score(y_test,preds)
print(f"정확도:{acc :.2f}%")

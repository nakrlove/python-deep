loss function 손실함수
cost function 비용함수
objective function 목적함수
error function 오차함수


import pandas as pd

df = pd.read_csv("https://bit.ly/wine_csv_data")
df

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X = data = df.iloc[:,:3]
y = target = df.iloc[:,3:]

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

sns.kdeplot(X[:,0])
plt.xlim(-2,2)

# 훈련/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234,stratify=y)
dt = DecisionTreeClassifier(criterion='gini',max_depth=3)
dt.fit(X_train,y_train)
preds = dt.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"정확도: {acc*100:.2f}%")


dt.predict(X_test)
dt.predict_proba(X_test)


# 의사결정 참조 자료
# http://naver.me/5Ocz9FuI
from sklearn.tree import plot_tree
plt.figure(figsize=(10,7))
#plot_tree(dt)
plot_tree(dt,max_depth=3,feature_names=['alcohol','sugar','pH'])
plt.show()



# 오후 4교시
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234,stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.7, random_state=1234)
# dt = DecisionTreeClassifier(criterion='gini', min_samples_leaf=2, min_samples_split= 10 )

len(X_train)
len(y_train)
len(X_val)
len(y_val)
len(X_test)
len(y_test)
  min_samples_split=2,
  min_samples_leaf=1,
dt = DecisionTreeClassifier(criterion='gini', max_depth=3)
dt = DecisionTreeClassifier( criterion="gini", min_samples_leaf=2,min_samples_split=10)
dt.fit(X_train, y_train)
preds = dt.predict(X_val)
acc = accuracy_score(y_val, preds)
print(f"정확도: {acc*100:.2f}%")

preds = dt.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"정확도: {acc*100:.2f}%")

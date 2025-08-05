
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



species를 target으로 해서
5가지 과정을 수행하여 정확도를 구해봅시다.

import seaborn as sns
df = sns.load_dataset('penguins')
df.info()
df
df =df.dropna()

df

X = df.iloc[:,2:6]
X
y = df['species']
#y = df.iloc[:,:1]
y

1. feature selection
2. 어떤 모델
3. 각 모델별로 옵션값,인수값, 하이퍼파라미터 값
최적의 하이퍼 파라미터를 찾아낸다.

kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(X, y)
kn._fit_X
kn._y
kn._fit_method

preds = kn.predicst(X)

acc = accuracy_score(y, preds)
acc
print(f"정확도:{acc :.2f}%")
print("정확도:", round(acc * 100, 2), "%")

# 2교시
import numpy as np
np.ones(shape=(1,35)).flatten()
np.array([1]*35)

y = target = label
y



#3교시
분류(classification) VS 균집(clustering)
classification, clustering



X = df.iloc[:,2:5]
X
y = df['species']
#y = df.iloc[:,:1]
y
np.shuffle()

from sklearn.model_selection import train_test_split

X_train, X_test,y_train, y_test = train_test_split( X,y, train_size=0.8)
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(X_train,y_train)

preds = kn.predict(X_test)
acc = accuracy_score(y_test,preds)
print(f"정확도:{acc :.2f}%")

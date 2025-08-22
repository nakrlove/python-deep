"""
차원 : 칼럼 = 속성  = 피쳐 의 갯수 ,
5차원 -> 2차원 : 피쳐 셀력션 featurre selection
PCA : 5차원 -> 2차원
"""



from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.decomposition import PCA



iris = load_iris()
X = iris['data']
y = iris['target']



X_train , x_test, y_train,y_test =  train_test_split(X,y,test_size=.3)


lr = LogisticRegression()
lr.fit(X_train,y_train)

y_pred = lr.predict(x_test)
accuracy_score(y_test,y_pred)

pca = PCA(n_components=2)
pca_X = pca.fit_transform(X)
# pca_X = pca.transform(X)
pca_X.shape
pca.inverse_transform(pca_X)

pca_X[:2]
pca.explained_variance_ratio_
X_train , x_test, y_train,y_test =  train_test_split(pca_X,y,test_size=.3)




#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import seaborn as sns
dia = sns.load_dataset("diamonds")
dia = dia.sample(frac = .1)

dia.info()

X = dia.iloc[: , [0,4,5,6,7,8,9]]
y = dia['cut']
X[:2]


X_train , x_test, y_train,y_test =  train_test_split(X,y,test_size=.3)


lr = LogisticRegression()
lr.fit(X_train,y_train)

y_pred = lr.predict(x_test)
accuracy_score(y_test,y_pred)


#++++++ PCA적용 변경
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


X_pca.shape
pca.inverse_transform(X_pca)


X_train , x_test, y_train,y_test =  train_test_split(X_pca,y,test_size=.3)
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(x_test)
accuracy_score(y_test,y_pred)


pca.inverse_transform(X_pca)
pca.explained_variance_ratio_


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
fruits = np.load("C:\\Users\\Admin\\study01\\20250811\\fruits_300.npy")
# fruits = fruits.reshape(300,10000)
fruits_2d = fruits.reshape(-1,100*100)
fruits_2d[:2]


#군집을 합니다.
pca = PCA(n_components=3,svd_solver='full')
X_pca = pca.fit_transform(X)

X_pca[:3]
y = target = np.array([0]*100 + [1]*100 + [2]*100)




X = fruits_2d
X_train , x_test, y_train,y_test =  train_test_split(X,y,test_size=.3)
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(x_test)
accuracy_score(y_test,y_pred)

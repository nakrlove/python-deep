import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv("https://bit.ly/wine_csv_data")

X = data  =  df.iloc[:, :3]
y = target = df.iloc[:, 3:]

scaler = StandardScaler()
scaler.fit
(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234, stratify=y)

# dt = DecisionTreeClassifier(criterion='gini', min_samples_leaf=2, min_samples_split= 10 )
dt = DecisionTreeClassifier(criterion='gini', max_depth=3)
dt.fit
(X_train, y_train)
preds = dt.predict(X_test)
acc = accuracy_score(y_test, preds)
print(acc) 




from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate


dt = DecisionTreeClassifier( criterion="gini", min_samples_leaf=2,min_samples_split=10)
dt.fit(X_train, y_train)
preds = dt.predict(X_val)
acc = accuracy_score(y_val, preds)

#======================================================
dt = DecisionTreeClassifier( criterion="gini", min_samples_leaf=2,min_samples_split=10)
score = cross_val_score(dt,X_train,y_train,cv=10,scoring="accuracy")

np.sum(score) / len(score) 


cross_validate(dt,X_train,y_train,cv=10,scoring="accuracy")
np.sum(test_score[])
#cross_val_score(
#    estimator,
#    X,
#    y=None,
#    *,
#    groups=None,
#    scoring=None,
#    cv=None,
#    n_jobs=None,
#    verbose=0,
#    params=None,
#    pre_dispatch="2*n_jobs",
#    error_score=np.nan)

print(f"정확도: {acc*100:.2f}%")


from sklearn.model_selection import GridSearchCV
param_grid = {'criterion':('gini','entropy'),'min_samples_leaf':[1,2,3,4,5],'min_samples_split':[5,10,15,20]}
dt_grid = GridSearchCV(dt,param_grid = param_grid,scoring='accuracy',cv=5)
dt_grid.fit(X_train,y_train)

#============================================
dt_grid.best_score_
dt_grid.best_params_
dt_best = dt_grid.best_estimator_
#============================================

preds = dt_best.predict(X_test)
accuracy_score(y_test,preds)



# iris , titanic, tips, gampminder , diamods ,mpg, wine-quality-red

GridSearchCV를 이용하여 최고의 모델과 그 모델의 최적의 파라미터를 찾아봅시다.

KNeighbors
DecisioTree
LogisticRegression()
RandomForest

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

df = sns.load_dataset('iris')
df.info() 
#1)의사결정 나무로 species를 예측해 봅시다.
#2)랜덤포레스트로 species 를 예측해 봅시다.정확도는 ?


X = data  =  df.iloc[:, :4]
y = target = df['species']
y = y.astype('category')

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234, stratify=y)

#===============================================================
# dt = DecisionTreeClassifier(criterion='gini', min_samples_leaf=2, min_samples_split= 10 )
dt = DecisionTreeClassifier(criterion='gini', max_depth=3)
#dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
preds = dt.predict(X_test)



y
acc = accuracy_score(y_test, preds)
print(f"DecisionTree 정확도: {acc*100:.2f}%")


rf = RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=3, min_samples_split=3, min_samples_leaf=2)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
rf.feature_importances_  # feature의 중요한 정도를 보여주는 속성
array([0.09672232, 0.0115251 , 0.42769184, 0.46406074])

df = pd.DataFrame(np.array([0.09672232, 0.0115251 , 0.42769184, 0.46406074]))
df = df.reset_index()
df.columns = ['feature','value']
# 시리즈로 만들어 인덱스를 붙인다
ser = pd.Series(rf.feature_importances_)

# 내림차순 정렬을 이용한다
top = ser.sort_values(ascending=False)

df.info()
df['feature'] = df['feature'].astype("category")
sns.barplot(data=df,x = 'feature' , y='value',hue='feature')




acc = accuracy_score(y_test, preds)
print(f"RandomForest 정확도: {acc*100:.2f}%")




"""
#======= 
 ExtraTreesClassifier
 GradientBoostingClassifier
 XGBClassifier 
 LGBMClassifier
====================================
"""
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
from lightgbm import LGBMClassifier

df = sns.load_dataset('iris')
df.info() 
#1)의사결정 나무로 species를 예측해 봅시다.
#2)랜덤포레스트로 species 를 예측해 봅시다.정확도는 ?


X = data  =  df.iloc[:, :4]
y = target = df['species']

"""
# 방법 1 
mapping = {'setosa':1,'versicolor':2 ,'virginica':3}
y.map(mapping)

# 방법 2 LabelEncoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


#방법 3 OneHotEncoding
#  'setosa' 'versicolor' 'virginica'
#    1           0            0
#    0           1            0
#    0           0            1

y = target = df[['species']]
encoder = OneHotEncoder()
y_encoded  = encoder.fit_transform(y)

y_encoded = pd.get_dummies(y) //위 OneHotEncoder한줄로 처리하는 방법
"""



scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#X_train, X_test, y_train, y_test = train_test_split(X, y_encoded)
X_train, X_test, y_train, y_test = train_test_split(X, y)

#rf = ExtraTreesClassifier(n_estimators=10, criterion='entropy', max_depth=3, min_samples_split=3, min_samples_leaf=2)
#rf = ExtraTreesClassifier()
#rf = GradientBoostingClassifier()

#rf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
rf = LGBMClassifier()

#cross_val_score(rf,X_train,y_train)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)

acc = accuracy_score(y_test, preds)
print(f"RandomForest 정확도: {acc*100:.2f}%")
rf.feature_importances_

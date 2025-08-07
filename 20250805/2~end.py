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
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234, stratify=y)

#===============================================================
# dt = DecisionTreeClassifier(criterion='gini', min_samples_leaf=2, min_samples_split= 10 )
dt = DecisionTreeClassifier(criterion='gini', max_depth=3)
dt.fit(X_train, y_train)
preds = dt.predict(X_test)
dt.predict(np.array([[ 1.9354021 , -0.68171895,  3.05706534]]))
acc = accuracy_score(y_test, preds)

print(acc) 
#===============================================================
dt = RandomForestClassifier()
dt.fit(X_train, y_train)
preds = dt.predict(X)
dt.predict(np.array([[ 1.9354021 , -0.68171895,  3.05706534]]))



df = sns.load_dataset('iris')
df.info() 
1)의사결정 나무로 species를 예측해 봅시다.
2)랜덤포레스트로 species 를 예측해 봅시다.정확도는 ?


X = data  =  df.iloc[:, :4]
y = target = df['species']


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

acc = accuracy_score(y_test, preds)
print(f"DecisionTree 정확도: {acc*100:.2f}%")


rf = RandomForestClassifier()
rf.fit(X_train, y_train)
preds = rf.predict(X_test)

acc = accuracy_score(y_test, preds)
print(f"RandomForest 정확도: {acc*100:.2f}%")


    
from sklearn.model_selection import cross_validate,cross_val_score
# CV
rf = RandomForestClassifier(n_estimators = 10, criterion='entropy')
cross_validate(rf, X_test, y =y_test, scoring='accuracy',cv=5)
score = cross_val_score(rf, X_test, y =y_test, scoring='accuracy',cv=5)
np.mean(score)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)

acc = accuracy_score(y_test, preds)
print(f"RandomForest 정확도: {acc*100:.2f}%")


#==========================================================
#ensemble : RandomForest
#여러모델들:
    
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
lr = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
kn = KNeighborsClassifier()
vo_clf = VotingClassifier(estimators=[('lr', lr),('dt',dt), ('rf', rf),('neighbor',kn)],voting='soft')
vo_clf.fit(X_train,y_train)
preds = vo_clf.predict(X_test)


accuracy_score(y_test,preds)    
precision_score(y_test,preds)
recall_score(y_test,preds)
f1_score(y_test,preds)
#precision recall

"""    
                             Acutal
                        
                     Positive     Negative
          Positive      5           10   
Predict   Negative     15           20

Precision = 
Recall    = 
F1 score  = 2* Precision*Recall /Precision + Recall = ?



Precision (정밀도) = TP / (TP + FP)
Precision = 5 / (5 + 10) 
            = 5 / 15 
            = 0.33333


Recall (재현율)     = TP / (TP + FN)
                       = 5 / (5 + 15) 
                       = 5 / 20 
                       = 0.25
F1 Score 
F1 = 2 × (Precision × Recall) /(Precision + Recall)
    = 2 x (0.33 X 0.25) / (0.33 + 0.25) 
    = 2 x 0.083 / 0.583
    = 0.2857
    
FP : Postive 라고 예측했는데 틀린것 :판사
FN :Negative 라고 예측했는데 틀린것 :의사
재현율 Recall    
    
"""    
#   위계산을 아래 함수로 대체함 
accuracy_score(y_test,preds)    
precision_score(y_test,preds)
recall_score(y_test,preds)
f1_score(y_test,preds)    
    
# 분류,회귀 ,군집,차원축소,: 머신러닝
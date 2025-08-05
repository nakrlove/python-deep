import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()

# 데이터 준비
X = df.copy()
y = np.array([1]*35 + [0]*14)
kn = KNeighborsClassifier()
kn.fit(X, y)
  
preds = kn.predict(X)

print(  f"정확도 : {accuracy_score(y,preds)} % 맞았습니다")
print(  f"정확도 : {(sum(y  == preds)/  len(y) )* 100} % 맞았습니다")

몇 개만 예측하고 싶다
kn.predict([[10.5 , 120.3]])



# 데이터준비
url = "https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/titanic.csv"
url = "https://raw.githubusercontent.com/zief0002/miniature-garbanzo/main/data/gapminder.csv"


titanic = pd.read_csv(url)
titanic.columns
titanic.info()


#KNeighborsClassifier() / K-최근접 이웃
#LogisticRegression() / 로지스틱 회귀
#DecisionTreeClassifier() / 결정 트리
#RandomForestClassifier() / 랜덤 포레스트

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression



X = titanic[['income','co2','population']]
X
y = titanic['life_exp'].astype('int')
y
kn = KNeighborsClassifier()

kn = KNeighborsRegressor()
kn.fit(X, y)
preds = kn.predict(X)
preds
y[:10] == preds

X = titanic[['SibSp','Parch']]
X
y = titanic['Survived']
y
kn = KNeighborsClassifier()
kn.fit(X, y)
preds = kn.predict(X)
preds
round(accuracy_score(y,preds) * 100,2)



X = df.copy()
y = np.array([1]*35 + [0]*14)


import seaborn as sns
df = sns.load_dataset('diamonds')
sns.get_dataset_names()
df.columns
X = df.iloc[: , :4]
#y = df.iloc[: , 4]
y = df['sex']
y

y = df['color']
X = df['']

df['color']



#각각 길이 무게 데이터가 담긴 df1과 df2를 하나의 df으로 만드려면
#df = pd.concat([df1, df2], ignore_index=True)
#pd.concat([df1, df2], axis=0)
#df = df1.concat(df2,ignore_index=True)
#pd.merge(df1, df2)

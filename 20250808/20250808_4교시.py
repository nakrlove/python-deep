#다중선형회귀 연습1
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score
from sklearn.datasets import load_iris
#데이터셋 준비
iris = load_iris()
X = iris['data'] 
y = iris['target']

X_train , X_test, y_train,y_test = train_test_split(X,y, test_size=.3,random_state=1234)

lr = LinearRegression()
lr.fit(X_train,y_train)
lr.coef_
lr.intercept_

y_pred = lr.predict(X_test)
y_true = y_test
mean_squared_error(y_true,y_pred)

#상호작용(교호작용) 고려하기 
poly = PolynomialFeatures(include_bias=False,interaction_only=True)
X = poly.fit_transform(X)
y = iris['target']

X_train , X_test, y_train,y_test = train_test_split(X,y, test_size=.3,random_state=1234)

lr = LinearRegression()
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)
y_true = y_test
mean_squared_error(y_true,y_pred)

X_test
lr.predict([[ 7.2 ,  3.2 ,  6.  ,  1.8 , 23.04, 43.2 , 12.96, 19.2 ,  5.76,10.8 ]])


#3차 상호작용(교호작용) 고려
poly = PolynomialFeatures(degree=3,include_bias=False,interaction_only=True)
X = poly.fit_transform(X)


#다중공선성 고려하기 : 상관관계 VIF값으로 보기
from statsmodels.stats.outliers_influence import variance_inflation_factor
#데이터셋 준비
iris = load_iris()
X = iris['data'] 
y = iris['target']

# 상관관계 VIF값으로 보기 값제거 --- (1)
variance_inflation_factor(X,0)
variance_inflation_factor(X,1)
variance_inflation_factor(X,2)
variance_inflation_factor(X,3)


np.corrceof()
X = X[:, :3]
X = X[:, 1:] #다중공선성이 큰것 제거

#상호작용(교호작용) 고려하기 
poly = PolynomialFeatures(include_bias=False,interaction_only=True)
X = poly.fit_transform(X)
y = iris['target']

X_train , X_test, y_train,y_test = train_test_split(X,y, test_size=.3,random_state=1234)

lr = LinearRegression()
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)
y_true = y_test
mean_squared_error(y_true,y_pred)




################## statsmodels ####################
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score
iris = load_iris()
iris['feature_names']
#iris['feature_names'] = ['sepal_length','sepal_width','peta_length','petal_width']
X = iris['data'] 
df = pd.DataFrame(X,columns=['sepal_length','sepal_width','petal_length','petal_width'])
df['species'] =  iris['target']


df_train , df_test = train_test_split(df, test_size=.3,random_state=1234)
model = sm.OLS.from_formula("species ~ sepal_length+sepal_width+petal_length+petal_width",data=df_train)


result = model.fit()
result.summary()
result.predict(df_test.iloc[:,:4])
y_pred = result.predict(df_test)
result.predict(df_test[:1])
r2_score(df_test['species'],y_pred)
mean_squared_error(df_test['species'], y_pred)




#상호작용값 interactin고려하기
model = sm.OLS.from_formula("species ~ sepal_length*sepal_width*petal_length*petal_width",data=df_train)
result = model.fit()
result.summary()
y_pred = result.predict(df_test)
mean_squared_error(df_test['species'], y_pred)

#상호작용값 interactin고려하기
model = sm.OLS.from_formula("species ~ sepal_length*sepal_width + petal_length*petal_width",data=df_train)
result = model.fit()
result.summary()
y_pred = result.predict(df_test)
mean_squared_error(df_test['species'], y_pred)


"""
# 다중공산성 고려하기
281.8044779767117
99.12012392513036
208.70366269134428
76.29110506984814

model = sm.OLS.from_formula("species ~ sepal_length*sepal_width + petal_length*petal_width",data=df_train)
model = sm.OLS.from_formula("species ~ sepal_width + petal_length*petal_width",data=df_train) <-다중공산성이 큰항목중(sepal_length)제거함
"""
for i in range(df_train.shape[1]-1):
   print( variance_inflation_factor(df_train,i)) # sepal_length 다중공산성이 큰항목 제외처리


url ="https://gist.githubusercontent.com/nnbphuong/def91b5553736764e8e08f6255390f37/raw/373a856a3c9c1119e34b344de9230ae2ea89569d/BostonHousing.csv"
df = pd.read_csv(url)
df.info()
X = df.iloc[:,:12] 
y = df.iloc[: ,[12]]



poly = PolynomialFeatures(include_bias=False,interaction_only=True)
X = poly.fit_transform(X)
X_train , X_test = train_test_split(df, test_size=.3,random_state=1234)

lr = LinearRegression()
lr.fit(X_train,y_train)
lr.coef_
lr.intercept_

y_true = lr.predict(X_test)
y_pred = y_true

mean_squared_error(y_true, y_pred)




# 상관관계 VIF값으로 보기 값제거 --- (1)
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
url ="https://gist.githubusercontent.com/nnbphuong/def91b5553736764e8e08f6255390f37/raw/373a856a3c9c1119e34b344de9230ae2ea89569d/BostonHousing.csv"
df = pd.read_csv(url)

"""
df_train,df_test =  train_test_split(df, test_size=.3,random_state=1234)
model = sm.OLS.from_formula("MEDV ~ (CRIM + ZN + INDUS + CHAS + NOX + RM + AGE)**2",data=df_train)

result = model.fit()
result.summary()

y_pred = result.predict(df_test)
mean_squared_error(df_test['species'], y_pred)
"""

# ======================================= #
# 다중공선성 확인을 위한 임시저장              #
# ======================================= #
vif_data = pd.DataFrame()
vif_data["Feature"] = df.columns
vif_data["VIF"] = [variance_inflation_factor(df.values,i) for i in range(df.shape[1])]


X = df.iloc[:,[0,1,2,3,4,6,7,8,9,11]]
y = df.iloc[:,[12]]
poly = PolynomialFeatures(include_bias=False,interaction_only=True)
X = poly.fit_transform(X)

X_train , X_test, y_train,y_test = train_test_split(X,y, test_size=.3,random_state=1234)

lr = LinearRegression()
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)
y_true = y_test
mean_squared_error(y_true,y_pred)

"""
사이킷런에서 이이리스 데이터를 가져와 훈련세트 테스트 세트로 나눈다
다중회귀분석
교호작용 , 다중공선성을 고려한다
변수선택은 양방향으로 한다.
최종 MSE를 출력한다.
statsmodels를 사용.
"""
#단순선형회귀
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
df = sns.load_dataset("iris")
df

#sepal_length값으로 sepal_width값을 마추고 싶음
df = df.iloc[: , 2:4]
df
sns.scatterplot(data=df , x='petal_length',y='petal_width')


"""
# 'petal_length':설명변수
# 'petal_width' : 반응변수로 하는 단순회귀분석을 합니다.
"""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = df[['petal_length']]
y = df[['petal_width']]
lr = LinearRegression()
lr.fit(X,y)


preds = lr.predict(X[:3])
all_preds = lr.predict(X)
#모든 데이터에 대한 잔차 제곱의 평균을 구해주세요
mse = mean_squared_error(y, all_preds)

"""
=============================================
#3개의 데이터에 대한 잔차를 구해주세요
잔차 = 실제값 - 예측값
      preds - y[:3]


=============================================      
#모든 데이터에 대한 잔차 제곱의 평균을 구해주세요
mean_squared_error(y, all_preds)   : MAE


      preds - lr.predict(X)
     = np.mean((preds - y)**2)  
     
=============================================     
#모든 데이터에 대한 잔차 절대값을 구해주세요     
mean_absolute_error(y, all_preds)  : MSE
residuals_abs = np.abs(all_preds - y)


"""

lr.coef_   #기울기
lr.intercept_  #y 절편

y = lr.coef_ * X + lr.intercept_
import matplotlib.pyplot as plt
plt.plot(X,y , color='blue')

import numpy as np

residuals_abs = np.abs(all_preds - y)


"""
    statsmodels 으로 회귀
"""
import statsmodels.api as sm

X = sm.add_constant(X)
model  = sm.OLS(y, X)

obj = model.fit()
obj.predict(X[:3])
obj.summary()

X = df[['petal_length']]

df = sns.load_dataset("iris")
#model = sm.OLS.from_formula("sepal_width ~ sepal_length",data=df)
model = sm.OLS.from_formula('petal_width ~ petal_length',data=df)
result = model.fit()
result.summary()
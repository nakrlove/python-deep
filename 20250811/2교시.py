import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
#데이터셋 준비

"""
    다중회귀분석을 이용하여
    train_test_split (7:3)하여
    mse를 제출해 주세요
"""
df = sns.load_dataset('diamonds')
df.info()


# y : price
y = df['price']

# X : 수치형 변수만 사용
X = df[['carat','depth']]
X.isnull().sum()

# VIF 계산 함수

# def calculate_vif(X):
#     vif_data = pd.DataFrame()
#     vif_data["Feature"] = X.columns
#     vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
#     return vif_data

X.shape
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# vif = calculate_vif(X)
# print(vif)


X_train , X_test, y_train,y_test = train_test_split(X,y, test_size=.3,random_state=1234)

lr = LinearRegression()
lr.fit(X_train,y_train)
lr.coef_
lr.intercept_

y_pred = lr.predict(X_test)
y_true = y_test
mean_squared_error(y_test,y_pred)






# =====================


import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# 데이터 로드
diamonds = sns.load_dataset('diamonds')

# 범주형 변수(cut, color, clarity)를 수치형으로 변환
label_encoders = {}
for column in ['cut', 'color', 'clarity']:
    le = LabelEncoder()
    diamonds[column] = le.fit_transform(diamonds[column])
    label_encoders[column] = le

# 독립변수(X)와 종속변수(y) 설정
X = diamonds[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']]
y = diamonds['price']

# 데이터 분할 (7:3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 다중회귀분석 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 테스트 데이터 예측
y_pred = model.predict(X_test)

# MSE 계산
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
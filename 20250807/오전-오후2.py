#다중선형회기
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

df = sns.load_dataset("diamonds")

df.info()
#X = df.iloc[: , [0,4,5,6,8,9]].corr()
X = df[['carat']]
#X = df.iloc[: , [0,4,5]]
sns.scatterplot(data=X , x='carat',y='price', hue='cut')

X.info()
X.corr()
y = df['price']
lr = LinearRegression()
lr.fit(X,y)

lr.predict(X.iloc[[53935]])


df.iloc[53935]['carat']

"""
preds = lr.predict(X.iloc[2])
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


print("정확도:", accuracy_score(y_test, preds))
#53935 2757
"""



"""

 2

"""

X = pd.read_csv('https://bit.ly/perch_csv_data')
y = perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

assert(len(X) != len(y))


X_train , X_test, y_train,y_test = train_test_split(X,y, test_size=.3)

poly = PolynomialFeatures(include_bias=False)
X_train_poly = poly.fit_transform(X_train)
poly.get_feature_names_out()
X_test_poly = poly.fit_transform(X_test)

lr = LinearRegression()
lr.fit(X_train_poly,y_train)
lr.predict(X_train_poly)


# ================#MSE 설명 20250808 ================
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score
mean_squared_error(y_test, lr.predict(X_test_poly))
mean_absolute_error(y_test, lr.predict(X_test_poly))
r2_score(y_test, lr.predict(X_test_poly))
# ================#MSE 설명 20250808 ================

np.mean((lr.predict(X_test_poly) - y_test) **2) #MSE
lr.score(X_test_poly,y_test)



# ================PolynomialFeatures 설명 20250808 ================
poly = PolynomialFeatures(include_bias=False)
poly.fit_transform(np.array([[1,2]]))
poly.fit_transform(np.array([[2,3]]))
poly.fit_transform(np.array([[3,4]]))
poly.fit_transform(np.array([[2,3,4]]))
poly.transform()
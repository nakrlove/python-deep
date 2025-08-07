import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

#pd.read_csv("https://gist.githubusercontent.com/rickiepark/2cd82455e985001542047d7d55d50630/raw/1152e11a3d792b23c5e4b1e202062f30ed2a702a/perch_length_weight.py")
X = perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
y = perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

plt.rcParams['font.family'] = 'Malgun Gothic'

sns.scatterplot(x=perch_length,y=perch_weight)
plt.xlabel('길이')
plt.ylabel('무게')
X = X.reshape(-1,1)
lr = LinearRegression()
lr.fit(X,y)
lr.predict(np.array([[33.0],[45],[25.1]]))

lr.coef_      #기울기
lr.intercept_ #절편


#### Scikit-learn으로 회귀분석
df = sns.load_dataset('iris')
df.info()
X = df[['sepal_length']]
y = df['sepal_width']

lr = LinearRegression()
lr.fit(X,y)
lr.predict(X[:2])

X[:2]
y[:2]


#### statsmodels로 회구분석
# statsmodels (통계학전문 프로그램) 통계적 모델링을 위한 강력한 기능을 제공
import statsmodels.api as sm
#x = data.a.values
#y = data.b.values
model = sm.OLS(y, X)

X = sm.add_constant(X)
result = model.fit()

model = sm.OLS(y, X)
sm.summary()
print(result.summary())


model = sm.OLS.from_formula("sepal_width ~ sepal_length",data=df)
result = model.fit()
result.summary()


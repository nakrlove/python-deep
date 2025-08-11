import pandas as pd 
import seaborn as sns 
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score
from local_test.rgrn import Regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

iris = load_iris()
X = iris['data'] 
y = iris['target']

# X = pd.DataFrame(X)
# X.corr() #상관관계 
# sns.heatmap(X.corr(),annot=True,cmap='coolwarm',linewidths=.2)
# plt.show()

#훈련세트/테스트세트
# 회귀
rg = Regression()
rg.commonRegress(X,y)
# lr.predict([[5.2, 4.1, 1.5, 0.1]])
# lr.predict([[6.9, 3.1, 5.1, 2.3]])
# lr.predict([[7.2, 3.2, 6. , 1.8]])
rg.predict([[7.2, 3.2, 6. , 1.8]])

#상호작용고려하기
poly = PolynomialFeatures(include_bias=False ,interaction_only=True)
X = poly.fit_transform(X)
y = iris['target']

rg.commonRegress(X,y)
rg.predict([[ 6.9 ,  3.1 ,  5.1 ,  2.3 , 21.39, 35.19, 15.87, 15.81,  7.13,11.73]])

"""
============================================================
다중공선성 고려하기
#다중공선성 고려하기 : 1) VIF값으로 보기 , 2) 상관관계 
============================================================
"""
#다중공선성 고려하기 : 1) VIF값으로 보기 , 2) 상관관계 
#데이터 가져오기
iris = load_iris()
X = iris['data'] 
y = iris['target']

#1) 다중공선성 제거를 위해 컬럼하나를 제거하고 테스트
X = X[: , 1:]
#상호작용고려하기
poly = PolynomialFeatures(include_bias=False ,interaction_only=True)
X = poly.fit_transform(X)
y = iris['target']

# 회귀
rg = Regression()
rg.commonRegress(X,y)

#다중공선성 값확인함 
variance_inflation_factor(X,0)
variance_inflation_factor(X,1)
variance_inflation_factor(X,2)
variance_inflation_factor(X,3)


#2) 위 다중공선성 대신 상관관계로 할수도 있다.
"""
상관계수는 두 변수 간 관계의 강도와 방향을 숫자로 나타낸 것이에요.
값은 -1부터 1까지 있을 수 있는데,
1에 가까울수록 두 변수는 거의 똑같은 방향으로 아주 밀접하게 움직임을 의미해요.
0에 가까우면 서로 관련이 거의 없고,
-1에 가까우면 완전히 반대 방향으로 움직인다는 뜻입니다.
"""
# df = pd.DataFrame(X)
# df.corr()

iris = load_iris()
X = iris['data'] 
y = iris['target']

X = X[: , :3]
#상호작용고려하기
poly = PolynomialFeatures(include_bias=False ,interaction_only=True)
X = poly.fit_transform(X)
y = iris['target']

# 회귀
rg = Regression()
rg.commonRegress(X,y)


"""
======== statsmodels ========
"""
iris = load_iris()
X = iris['data'] 
y = iris['target']
iris['feature_names'] = ['sepal_length','sepal_width','petal_length','petal_width']
df = pd.DataFrame(X,columns=iris['feature_names'])
df['species'] = iris['target']

df_train , df_test = train_test_split(df, test_size=.3,random_state=1234)
model = sm.OLS.from_formula("species ~ sepal_length+sepal_width+petal_length+petal_width",data=df_train)

#상호작용 
model = sm.OLS.from_formula("species ~ sepal_length*sepal_width*petal_length*petal_width",data=df_train)
#2개끼리 상호작용 
model = sm.OLS.from_formula("species ~ sepal_length*sepal_width+petal_length*petal_width",data=df_train)


result = model.fit()
result.summary()
result.predict(df_test.iloc[:,:4])

y_pred = result.predict(df_test)
result.predict(df_test[:1])
r2_score(df_test['species'],y_pred)
mean_squared_error(df_test['species'], y_pred)

#다중공선성 값확인함 
variance_inflation_factor(df_train,0)
variance_inflation_factor(df_train,1)
variance_inflation_factor(df_train,2)
variance_inflation_factor(df_train,3)

for i in range(df_train.shape[1]-1):
    print(variance_inflation_factor(df_train,i))



# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# target = raw_df.values[1::2, 2]

data_url ="https://gist.githubusercontent.com/nnbphuong/def91b5553736764e8e08f6255390f37/raw/373a856a3c9c1119e34b344de9230ae2ea89569d/BostonHousing.csv"
df = pd.read_csv(data_url)
df.info()
X = df.iloc[: , :12]
y = df.iloc[: , 12]

# X = iris['data'] 
# y = iris['target']

# 회귀
# rg = Regression()
# rg.commonRegress(X,y)
# rl = rg.LRInfo()

vif_data = pd.DataFrame()
vif_data["Feature"] = df.columns
vif_data["VIF"] = [variance_inflation_factor(df.values,i) for i in range(df.shape[1])]

X = df.iloc[: , [0,1,2,3,4,6,7,8,9,11]]
y = df.iloc[: , 12]

#상호작용고려하기
poly = PolynomialFeatures(include_bias=False ,interaction_only=True)
X = poly.fit_transform(X)

rg = Regression()
rg.commonRegress(X,y)
rl = rg.LRInfo()



"""
======== statsmodels ========
"""

data_url ="https://gist.githubusercontent.com/nnbphuong/def91b5553736764e8e08f6255390f37/raw/373a856a3c9c1119e34b344de9230ae2ea89569d/BostonHousing.csv"
df = pd.read_csv(data_url)
rg = Regression()

# param = "MEDV ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + LSTAT"
param1 = "MEDV ~ (CRIM + ZN + INDUS + CHAS + NOX + RM + AGE)**2"
summary , res = rg.trainTestSplit(df ,param1)
summary.summary()
res
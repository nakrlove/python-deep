#문제를 풀어보세요
# diamonds 데이터셋의 'price'를 잘 맞추는 다중선형회귀분석을 해 봅시다.
#(test_set : MSE 가 가장 작아지도록)
import seaborn as sns
#다중선형회귀 연습1
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score


df = sns.load_dataset('diamonds')
df = df.dropna()
df.isna().sum()
DataFrame
df.info()

X = df.iloc[: , [0,4,5,7,8]]
y = df['price']
# 2. 훈련/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#상호작용(교호작용) 고려하기
lr = LinearRegression()
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)
y_true = y_test
mean_squared_error(y_true,y_pred)



from statsmodels.stats.outliers_influence import variance_inflation_factor
df.shape



vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]



X_reduced = X.drop(columns=["y","x", "table"])  # VIF 가장 큰 변수 제거

vif_data = pd.DataFrame()
vif_data["Feature"] = X_reduced.columns
vif_data["VIF"] = [variance_inflation_factor(X_reduced.values,i) for i in range(X_reduced.shape[1])]





for i in range(df.shape[1]):
    df.isna().sum()
    print(df.values)
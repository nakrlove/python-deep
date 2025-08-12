import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge,Lasso, ElasticNet
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

# ridge = Ridge()
alpha_list = [ 0.01 ,0.1 , 0.2,0.3,1,2,10]
# ridge = Ridge(alpha=0.01)
for alpha in alpha_list:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X,y)
    y_pred = ridge.predict(X)
    print(mean_squared_error(y,y_pred))


    
    ridge.coef_

# Lasso
for alpha in alpha_list:
    ridge = Lasso(alpha=alpha)
    ridge.fit(X,y)
    y_pred = ridge.predict(X)
    print(mean_squared_error(y,y_pred))    

# l1_list = [ 0.01 ,0.1 , 0.2,0.3,1,2,10]
l1_list = [0.01, 0.1, 0.2, 0.3, 1.0]  # 2, 10 제거
for alpha in alpha_list:
    for li in l1_list:
        ridge = ElasticNet(alpha=alpha,l1_ratio=li)
        ridge.fit(X,y)
        y_pred = ridge.predict(X)
        print(f"{alpha} : {li} 일때 {mean_squared_error(y,y_pred)}")  
print()


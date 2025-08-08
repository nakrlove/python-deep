import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score

vif_data = pd.DataFrame()
vif_data["Feature"] = df.columns
vif_data["VIF"] = [variance_inflation_factor(data.values,i) for i in range(df.shape[1])]


X = df.iloc[:,[0,1,2,3,4,6,7,8,9,11]]
y = df.iloc[:,[12]]

iris = load_iris()
iris['feature_names']
#iris['feature_names'] = ['sepal_length','sepal_width','peta_length','petal_width']
X = iris['data'] 
df = pd.DataFrame(X,columns=['sepal_length','sepal_width','petal_length','petal_width'])
df['species'] =  iris['target']



#문제를 풀어보세요
# diamonds 데이터셋의 'price'를 잘 맞추는 다중선형회귀분석을 해 봅시다.
#(test_set : MSE 가 가장 작아지도록)
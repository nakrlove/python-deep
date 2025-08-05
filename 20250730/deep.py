bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]



import pandas as pd
import numpy as np
arr = np.array([bream_length,bream_weight])
df = pd.DataFrame(arr.T)
# df.columns["length","weight"]
df.columns = ["length", "weight"]

df[:2]

df1 = pd.DataFrame({'length':bream_length, 'weight':bream_weight})
df


import matplotlib.pyplot as plt

plt.scatter(bream_length, bream_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

import seaborn as sns
sns.scatterplot(x='length',y='weight',data=df)
plt.show()




# 정리함 
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

df2 = pd.DataFrame(  {"length": smelt_length,  "weight": smelt_weight })
df2[:2]

df = pd.concat([df1, df2], ignore_index=True)
df

# df.values
df.to_numpy()
y = np.array([1]*35 + [0]*14)
target = y    # y를 target 이라고도 한다
X = df.to_numpy()
X

X.shape
X[0, :].shape


# classify : 분류한다
# classification : 분류
# classifier
# 1. 데이터 준비
# 2. 모델 가져오기
# 3. 학습시킨다  - 적합한다
# 4. 예측한다
# 5. 평가한다
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(X, y)
  
preds = kn.predict(X)
print(  f"{(sum(y  == preds)/  len(y) )* 100} % 맞았습니다")

plt.scatter(bream_length,bream_weight)
plt.scatter(smelt_length,smelt_weight)
plt.scatter(30,600,marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
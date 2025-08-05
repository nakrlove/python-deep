# 1. 데이터준비
#  - 결측치
#  - 데이터타입(str -> int,float)
#  - merge, 칼럼 선택
#  - feature selection,feature extraction
#  - 스케일링
#  - 데이터분포 (균형,불균형:데이터 빈도수)
#  - 훈련세트,테스트세트 분리(train_test_split)
#  - 훈련세트,검증세트 , 테스트세트 분리(train_test_split)
 
#  2. 모델 만들기
#      - 문제에 맞는 모델선택
#      - 여러가지 모델
#      - 모델 내부 알고리즘 살펴보기
#      - 앙상블
#  3. 학습
#      - fit()
#      - 몇가지 하이퍼파라미터 조정(GridSearchCV)
#      - 학습 결과물인 객체(=가중치)를 살펴보기 GPU , dt.class_,rf.bestestimator_,rf.feature_

#  4. 예측(추론)
#      - predict() : inference하기 추론칩
#      - 예측한 과정 들여다 보기(확률 > 0.5) predict_proba
#  5. 평가
#    - accuracy
#    - 정확도
#    - recall 
#    - precision
#    - F1 score
#    - ROC
#    - AUC
   
   
import pandas as pd
from sklearn.model_selection import train_test_split
# 가상의 사람 데이터

df = pd.DataFrame({
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'score':        [20, 25, 30, 35, 40, 45, 50, 60, 70, 80]
})

X = df['study_hours']
y = df['score']
X_train, X_test,y_train, y_test = train_test_split( X,y,  test_size=0.2,random_state=42)
print("훈련 데이터 개수:", len(X_train))
print("테스트 데이터 개수:", len(X_test))








import seaborn as sns
import pandas as pd
import numpy as py

df = sns.load_dataset('penguins')
df.describe()

#결측치 제거
df =df.dropna()

df.info()
X = df.iloc[:,2:6]
y = df['species']

#표준화
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier #확률적 경사하강법
from sklearn.tree import DecisionTreeClassifier#결정 트리
from sklearn.metrics import accuracy_score
scaler = StandardScaler()


#X_train, X_test,y_train, y_test = train_test_split( X,y, train_size=0.7 ,stratify=y
X_train, X_test,y_train, y_test = train_test_split( X,y, train_size=0.8)
#kn = LogisticRegression()
#kn = KNeighborsClassifier()
#kn = SGDClassifier(loss = 'log_loss')
kn = SGDClassifier(loss = 'log_loss',max_iter=10) #max_iter 반복
epoch = iteration #max_iter와 반복 


kn = DecisionTreeClassifier()
kn = DecisionTreeClassifier(criterion='entropy',max_depth=2,min_samples_split=5,min_samples_leaf=2)
kn.fit(X_train,y_train)


preds = kn.predict(X_test)
acc = accuracy_score(y_test,preds)
print(f"정확도:{acc :.2f}%")





# 교차검증과 그리드서치




















sns.scatterplot(data=X,)

import matplotlib.pyplot as plt

x = np.linsspace(-3,3)
plt.scatter(x,X[:0], color='blue' maker='o')


kn = KNeighborsClassifier()
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(X, y)

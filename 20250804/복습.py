#연습
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1) 데이터 준비
data = {
    'weather': ['sunny', 'sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny', 'sunny'],
    'humidity': [80, 65, 90, 70, 80, 75, 60, 85, 60, 72],
    'game': [False, True, False, True, True, True, True, True, True, False]
}

df = pd.DataFrame(data)

# 2) 범주형 변수 숫자로 변환
le = LabelEncoder()
df['weather_enc'] = le.fit_transform(df['weather'])  # sunny=2, overcast=1, rainy=0 등 숫자로 변환

# 3) 특징과 타겟 분리
X = df[['weather_enc', 'humidity']]
y = df['game']

# 4) 모델 생성 및 학습
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

# 5) 트리 시각화
plt.figure(figsize=(10,6))
plot_tree(clf, feature_names=['weather_enc', 'humidity'], class_names=['Game Off', 'Game On'], filled=True)
plt.show()

# 6) 새 데이터 예측 예시
new_data = pd.DataFrame({
    'weather_enc': le.transform(['sunny', 'rainy']),
    'humidity': [75, 80]
})
pred = clf.predict(new_data)
print("예측 결과:", pred) 





#-=====================================================
data = {
        'weather': ['sunny', 'sunny', 'rainy', 'overcast', 'sunny', 'rainy'],
        'temperature': [32, 28, 25, 22, 30, 35],
        'ac_on' : [True, False, False, False, True, False],
}

df = pd.DataFrame(data)

le = LabelEncoder()
df['weather_enc1'] = le.fit_transform(df['weather'])

X = df.iloc[:,:2]
X['weather'] =df['weather_enc1']
X.info()

#X = df[['weather_enc1', 'temperature']]
y = df.iloc[:,2]
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X, y)
new_data = pd.DataFrame({
    'weather': le.transform(['sunny', 'rainy']),
    'temperature': [45, 20]
})
pred = dt.predict(new_data)
print("예측 결과:", pred) 

#========================================================
data = {
    'time': ['morning', 'afternoon', 'evening', 'morning', 'evening', 'afternoon', 'morning', 'evening'],
    'temperature': [24, 30, 28, 22, 27, 33, 20, 26],
    'humidity': [40, 60, 55, 35, 50, 70, 30, 45],
    'ac_on': [False, True, True, False, True, True, False, False]
}


df = pd.DataFrame(data)
le = LabelEncoder()
df['time_enc'] = le.fit_transform(df['time']) 
df
X = df.iloc[:,[-1,1,2]]
y = df.iloc[:,3]

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X, y)
new_data = pd.DataFrame({
    'time_enc': le.transform(['afternoon']),
    'temperature': [29],
    'humidity': [65]
})
pred = dt.predict(new_data)
print("예측 결과:", pred) 



#====================================================
data = {
        "weather":['sunny','rainy', 'overcast','sunny', 'rainy', 'sunny','overcast', 'rainy'],
        'temperature': [30, 22, 25, 35, 20, 28, 24, 21],
        'time' : ['morning', 'afternoon', 'morning', 'afternoon', 'morning', 'evening', 'afternoon', 'evening'],
        'drink': ['iced coffee', 'hot tea', 'latte', 'iced coffee', 'hot tea', 'latte', 'iced coffee', 'hot tea'],
        }


df = pd.DataFrame(data)
weather_le = LabelEncoder()
time_le = LabelEncoder()
df['weather_enc'] = weather_le.fit_transform(df['weather']) 
df['time_enc'] = time_le.fit_transform(df['time']) 
X = df[['weather_enc','temperature','time_enc']]
y = df['drink']

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X, y)

new_data =  pd.DataFrame({
    "weather":weather_le.transform(['sunny']),
    "temperature":29, 
    "time" :time_le.transform(['afternoon']),
})
pred = dt.predict(new_data)
print("예측 결과:", pred[0])

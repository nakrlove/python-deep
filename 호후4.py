

from sklearn.model_selection import train_test_split

import seaborn as sns
df = sns.load_dataset('penguins')

#전처리 : 데이터타입 바꾸기, 결측치제거 ,정규화
#전처리
df = df.dropna()

X = df.iloc[:,2:5]
X
y = df['species']

X_train, X_test,y_train, y_test = train_test_split( X,y, train_size=0.8)
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(X_train,y_train)

preds = kn.predict(X_test)
acc = accuracy_score(y_test,preds)
print(f"정확도:{acc :.2f}%")

df[:2]
df['body_mass_g']



#정규화 :  0 ~ 1 사이 바꾸기
#X - 제일작은숫자(min) / max - min

def normolize(x) :
    return (x - x.min()) / (x.max() - x.min())


X = X.apply(normolize , axis=0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(X)
X = scaler.transform(X)


X = scaler.fit_transform(X)

#표준화 : 평균과 표준편차로 분포를 만들기
def standardize(x):
    return (x - x.mean()) / x.std()

X = X.apply(standardize , axis=0)
X

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X)
X = scaler.transform(X)


X = scaler.fit_transform(X)



#분류모델 : LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import seaborn as sns
df = sns.load_dataset('penguins')


#전처리 : 데이터타입 바꾸기, 결측치제거 ,정규화
#결측치제거 
df = df.dropna()

X = df.iloc[:,2:5]


y = df['species']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)



X_train, X_test,y_train, y_test = train_test_split( X,y, train_size=0.8)
kn = LogisticRegression()
kn = KNeighborsClassifier()
kn.fit(X_train,y_train)

preds = kn.predict(X_test)
acc = accuracy_score(y_test,preds)
print(f"정확도:{acc :.2f}%")


kn.predict(X_test[:1])
kn.predict_proba(X_test[10:30])



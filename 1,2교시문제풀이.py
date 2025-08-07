


# ===================== 문제풀이 ==============================================

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
df = sns.load_dataset("titanic")
df.info()


df = df.dropna()
X = df.iloc[:,[1,2,3,6]]
X

 = 

y = df['survived']
y


X_encoded = pd.get_dummies(X, drop_first=True) 


scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# 7. 예측 및 평가
y_pred = model.predict(X_test)
print("정확도:", accuracy_score(y_test, y_pred))
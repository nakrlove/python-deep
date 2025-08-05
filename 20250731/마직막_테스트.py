
#문제
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# 1. 데이터 불러오기
df = sns.load_dataset('penguins').dropna()

df = df.dropna()

X = df[['body_mass_g']]
df.info()

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

df['body_mass_g_scaled'] = X_scaled
df
kn.predict_proba(X_scaled)



# 자습
    # 전처리 → 벡터화 → 모델 로드 → 예측
    model = load_model('spam_model.pkl')
    vector = text_to_vector(text)
    prob = model.predict_proba(vector)[0][1]
    return '스팸입니다' if prob > 0.5 else '정상 메일입니다'




# 필요한 라이브러리
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. 샘플 데이터 (간단한 메시지들)
data = {
    'message': [
        'Free entry in 2 a wkly comp', 
        'Hey, how are you?',
        'WINNER! You have won a prize',
        'Call me when you can',
        'Claim your free vacation now',
        'Are we still meeting tomorrow?',
        'Limited time offer! Buy now',
        'Let’s catch up soon'
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = ham(일반 메시지)
}

df = pd.DataFrame(data)

# 2. 벡터화 (텍스트를 숫자로 변환)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# 3. 학습/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4. 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. 테스트 정확도 확인
y_pred = model.predict(X_test)
print("정확도:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

# 6. 새로운 메시지 예측 함수
def predict_spam(message):
    vec = vectorizer.transform([message])
    prediction = model.predict(vec)[0]
    return "스팸 메시지입니다." if prediction == 1 else "정상 메시지입니다."

# 7. 테스트
test_msg = "Free tickets just for you"
print("메시지:", test_msg)
print("예측 결과:", predict_spam(test_msg))


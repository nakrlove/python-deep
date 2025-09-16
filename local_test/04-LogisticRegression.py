import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('./local_test/youtube/LogisticRegressionData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
classifier.predict([[6]])
classifier.predict_proba([[6]]) # 합격할 확률 출력
classifier.predict([[4]])
classifier.predict_proba([[4]]) # 합격할 확률 출력



y_pred = classifier.predict(X_test)
y_pred # 예측 값
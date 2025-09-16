from tensorflow import keras
import numpy as np

(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

X_train = X_train.reshape(-1,28*28)
X_test = X_test.reshape(-1,28*28)

X_train.shape
X_test.shape

# inputs = keras.layers.Input(shape=(X_train.shape[0],))
inputs = keras.layers.Input(shape=(784,))
dense = keras.layers.Dense(10,activation ="softmax")
model = keras.Sequential([inputs,dense])
model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train)

# X_test = X_test.reshape(-1,28*28)
y_pred = model.predict(X_test)
result = np.argmax(y_pred)
import numpy as np
np.round(y_pred[2])

result = np.argmax(y_pred,axis=1)
result.shape

np.sum(y_test == result)/10000
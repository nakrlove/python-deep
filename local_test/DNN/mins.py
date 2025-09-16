from tensorflow import keras

(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
print(X_train.shape)
y_train.shape
X_test.shape
y_test.shape

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,10,figsize=(10,10))
for i in range(10):
    ax[i].imshow(X_train[i],cmap="gray")
plt.show()

X_train[0]



import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Physical devices:", tf.config.list_physical_devices())
print("GPU devices:", tf.config.list_physical_devices("GPU"))


X_train = X_train.reshape(60000,28*28)
X_train.shape

# inputs = keras.layers.Input(shape=(X_train.shape[0],))
inputs = keras.layers.Input(shape=(784,))
dense = keras.layers.Dense(10,activation ="softmax")
model = keras.Sequential([inputs,dense])
model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train)

X_test = X_test.reshape(-1,28*28)
y_pred = model.predict(X_test)

import numpy as np
np.round(y_pred[2])

np.argmax(y_pred[0])
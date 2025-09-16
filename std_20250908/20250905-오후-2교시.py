from keras.layers import Input, Conv2D,MaxPooling2D,Flatten,Dense,Dropout,Rescaling

import keras
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

# 1. 데이터 준비
# (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
# 폴더별로 라벨을 정리해서 넣는다
import os
from pathlib  import Path
train_dir = Path('./20250908/cats_and_dogs/Train')
validation_dir = Path('./20250908/cats_and_dogs/validation')
test_dir = Path('./20250908/cats_and_dogs/Test')

print('훈련용 고양이 이미지 전체 개수:',len(os.listdir(train_dir/'cats')))
print('훈련용 고양이 이미지 전체 개수:',len(os.listdir(train_dir/'dogs')))

print('훈련용 고양이 이미지 전체 개수:',len(os.listdir(validation_dir/'cats')))
print('훈련용 고양이 이미지 전체 개수:',len(os.listdir(validation_dir/'dogs')))

print('훈련용 고양이 이미지 전체 개수:',len(os.listdir(test_dir/'cats')))
print('훈련용 고양이 이미지 전체 개수:',len(os.listdir(test_dir/'dogs')))

#데이터셋 만들기 
# X_train = x_train.reshape(-1,28,28,1) / 255.0
# x_test  = x_test.reshape(-1,28,28,1) / 255.0
from keras.utils import image_dataset_from_directory

train_dataset = image_dataset_from_directory(
                    directory= train_dir,
                    # label_mode='categorical', #원핫인코딩으로 바꾸기
                    batch_size=32,
                    image_size=(180, 180)
                    )

validation_dataset = image_dataset_from_directory(
                        directory= validation_dir,
                        batch_size=32,
                        image_size=(180, 180)
                        )

test_dataset = image_dataset_from_directory(
                    directory= test_dir,
                    # label_mode='categorical', #원핫인코딩으로 바꾸기
                    batch_size=32,
                    image_size=(180, 180)
                    )


# 모델 만들기 
model = keras.Sequential([
                        Input(shape=(180,180,3)),
                        Conv2D(filters=32, kernel_size=3, activation='relu', padding='same',name='conv1' ),
                        MaxPooling2D(),
                        Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',name='conv2'),
                        MaxPooling2D(),
                        Conv2D(filters=128, kernel_size=3, activation='relu', padding='same',name='conv3'),
                        MaxPooling2D(),
                        Conv2D(filters=256, kernel_size=3, activation='relu', padding='same',name='conv4'),
                        MaxPooling2D(),
                        Flatten(),
                        Dense(100, activation='relu', name='my_dense_1'),
                        Dropout(0.4), 
                        Dense(1, activation='sigmoid'),    
                        ])
model.summary()

# model.compile(optimizer="Adam",loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.compile(optimizer="Adam",loss="binary_crossentropy", metrics=['accuracy'])
model.fit(train_dataset,epochs=5)

model.evaluate(test_dataset)
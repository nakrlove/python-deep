from keras.layers import Input, Conv2D,MaxPooling2D,Flatten,Dense,Dropout,Rescaling

import keras
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

import os
from pathlib  import Path
train_dir = Path('./20250908/apples/Train')
test_dir = Path('./20250908/apples/Test')

train_dir.exists()

from keras.utils import image_dataset_from_directory

train_dataset = image_dataset_from_directory(
                    directory= train_dir,
                    label_mode='categorical', #원핫인코딩으로 바꾸기
                    batch_size=32,
                    image_size=(180, 180)
                    )

# validation_dataset = image_dataset_from_directory(
#                         directory= validation_dir,
#                         # label_mode='categorical', #원핫인코딩으로 바꾸기
#                         batch_size=32,
#                         image_size=(180, 180)
#                         )

test_dataset = image_dataset_from_directory(
                    directory= test_dir,
                    label_mode='categorical', #원핫인코딩으로 바꾸기
                    batch_size=32,
                    image_size=(180, 180)
                    )


# 데이터 증강
from keras.layers import RandomRotation,RandomTranslation,RandomZoom,RandomFlip
data_augmentation = keras.Sequential([ Rescaling(1/255.0),
                                     RandomRotation(45/360, fill_mode='nearest'), # rotation_range=45에 해당
                                     # width_shift_range=0.2와 height_shift_range=0.2에 해당
                                     RandomTranslation(height_factor=0.2, width_factor=0.2, fill_mode='nearest'),
                                     RandomZoom(height_factor=0.2, fill_mode='nearest'), # zoom_range=0.2에 해당
                                     RandomFlip("horizontal"), # horizontal_flip=True에 해당
                                    ])
train_dataset.map(lambda x, y: (data_augmentation(x, training = True ), y))



# 모델 만들기 
model = keras.Sequential([
                        Input(shape=(180,180,3)),
                        Rescaling(1/255.0),
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
                        Dense(4, activation='softmax'),    
                        ])


# model.summary
# 컴파일
model.compile(optimizer="RmsProp",loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_dataset,epochs=5,)

# import sys, os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from std_20250908.imageTonumpy import load_cimg


import os
current_dir = os.getcwd()   # 현재 실행 위치
file_path = f"{current_dir}/std_20250908/120.jpg"  # 안전하게 경로 붙이기
print(file_path)

classes = model.predict(load_cimg(path=file_path,target_size=(180,180)))

print("fredict",classes)
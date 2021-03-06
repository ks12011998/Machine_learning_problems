# -*- coding: utf-8 -*-
"""identify_apparel.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nClnHVjaSyezmvJur50bc0I4E9K1IgAl
"""

import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt

! unzip /content/drive/My\ Drive/Identify_apparel.zip

train = pd.read_csv('/content/Identify_apparel/train_dataset/train.csv')
train.head()

test = pd.read_csv('/content/Identify_apparel/test_dataset/test.csv')
test.head()



import os
os.listdir('/content/Identify_apparel/test_dataset')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

train
for i in range(len(train['id'])):
    train['id'][i] = str(train['id'][i]) + '.png'

img_dir = '/content/Identify_apparel/train_dataset/train/'

train.head()

train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img(img_dir+train['id'][i], target_size=(28,28,3), grayscale=False)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)

y=train['label'].values
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20,validation_steps=2000,steps_per_epoch=8000, validation_data=(X_test, y_test))

test.head()
label = test['id']

test_img_dir = '/content/Identify_apparel/test_dataset/test/'

test_t = pd.read_csv('/content/Identify_apparel/test_dataset/test.csv')

test_t.head()
label = test_t['id']
label

test = pd.read_csv('/content/Identify_apparel/test_dataset/test.csv')
label = test['id']

for i in range(len(test['id'])):
  test['id'][i] = str(test['id'][i])+'.png'

test.head()

label

test_image = []
for i in tqdm(range(test.shape[0])):
    img = image.load_img(test_img_dir+test['id'][i], target_size=(28,28,3), grayscale=False)
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
test = np.array(test_image)

prediction = model.predict_classes(test)

prediction

submissions = pd.DataFrame({
         'id' : label,
         'label' : prediction
        })
submissions.to_csv('final_classification.csv',index=False)
submissions.head()
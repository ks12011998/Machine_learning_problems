
import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random

train = pd.read_csv('/home/suman/Important/Analytics_vidya_problems/Age_detection/train_dataset/train.csv')
train.head()

test = pd.read_csv('/home/suman/Important/Analytics_vidya_problems/Age_detection/test_dataset/test.csv')
test.head()

valid_imgs = random.sample(train['ID'].tolist(),2000)
train_imgs = list(set(train['ID'].tolist()) - set(valid_imgs))

os.mkdir('/home/suman/Important/Analytics_vidya_problems/Age_detection/train_dataset/train')
os.mkdir('/home/suman/Important/Analytics_vidya_problems/Age_detection/train_dataset/valid')

img_dir = '/home/suman/Important/Analytics_vidya_problems/Age_detection/train_dataset/Train/'


import shutil
for i in range(len(train_imgs)):
    shutil.copy(img_dir+train_imgs[i],'/home/suman/Important/Analytics_vidya_problems/Age_detection/train_dataset/train/')
    
for j in range(len(valid_imgs)):
    shutil.copy(img_dir+valid_imgs[j],'/home/suman/Important/Analytics_vidya_problems/Age_detection/train_dataset/valid/')
print (len(os.listdir('/home/suman/Important/Analytics_vidya_problems/Age_detection/train_dataset/train/')))
print (len(os.listdir('/home/suman/Important/Analytics_vidya_problems/Age_detection/train_dataset/valid/')))


train_df = pd.DataFrame(columns=['ID'])
train_imgs = os.listdir('/home/suman/Important/Analytics_vidya_problems/Age_detection/train_dataset/train/')
train_df['ID'] = train_imgs
train_df = pd.merge(train_df,train)
print (train_df.shape)
train_df.head()


valid_df = pd.DataFrame(columns=['ID'])
valid_imgs = os.listdir('/home/suman/Important/Analytics_vidya_problems/Age_detection/train_dataset/valid/')
valid_df['ID'] = valid_imgs
valid_df = pd.merge(valid_df,train)
print (valid_df.shape)
valid_df.head()

#importing the keras libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initailising the CNN
classifier = Sequential()
#Step1-Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation = 'relu'))

#Step2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Flattening
classifier.add(Flatten())

#Full connection
classifier.add(Dense(output_dim = 128,activation = 'relu'))
classifier.add(Dense(output_dim = 3,activation = 'softmax'))

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy' , metrics = ['accuracy'])


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1./255)



# Flow training images in batches of 20 using train_datagen generator
train_generator_df = train_datagen.flow_from_dataframe(dataframe=train_df, 
                                                       directory='/home/suman/Important/Analytics_vidya_problems/Age_detection/train_dataset/train/', 
                                                       x_col="ID", y_col="Class", class_mode='categorical', target_size=(64,64), batch_size=32)  

# Flow validation images in batches of 20 using test_datagen generator
validation_generator_df =  valid_datagen.flow_from_dataframe( dataframe=valid_df, 
                                                            directory='/home/suman/Important/Analytics_vidya_problems/Age_detection/train_dataset/valid/', 
                                                            x_col="ID", y_col="Class", 
                                                            class_mode='categorical', target_size=(64,64), batch_size=32)

import tensorflow as tf
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_loss')<0.1):
      print("\nReached 0.50 loss so cancelling training!")
      self.model.stop_training = True
        
callbacks = myCallback()


classifier.fit_generator(
            train_generator_df,
            validation_data = validation_generator_df,
           steps_per_epoch = 8000,
            epochs = 10,
             validation_steps = 2000,
            verbose = 2,
            callbacks = [callbacks])

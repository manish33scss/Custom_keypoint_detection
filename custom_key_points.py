import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, metrics, optimizers, models
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout
from sklearn import preprocessing
from matplotlib import pyplot as plt

modelx = Sequential()
input_shape=(224,224,3)
modelx.add(Conv2D(32, kernel_size=3, strides=(1,1), activation='relu', input_shape=input_shape))
modelx.add(MaxPooling2D(pool_size=(2, 2)))
modelx.add(Conv2D(64, kernel_size=3, strides=(1,1), activation='relu'))
modelx.add(MaxPooling2D(pool_size=(2, 2)))
modelx.add(Conv2D(128, kernel_size=3, strides=(1,1), activation='relu'))
modelx.add(MaxPooling2D(pool_size=(2, 2)))
modelx.add(Conv2D(256, kernel_size=3, strides=(1,1), activation='relu'))
modelx.add(MaxPooling2D(pool_size=(2, 2)))
modelx.add(Conv2D(512, kernel_size=3, strides=(1,1), activation='relu'))
modelx.add(Dropout(rate=0.2))
modelx.add(Flatten())
modelx.add(Dense(128, activation='relu'))
modelx.add(Dense(128, activation='relu'))
modelx.add(Dropout(rate=0.2))
modelx.add(Dense(2, activation='linear'))
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
modelx.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])

modelx.summary()



modelx.load_weights("/content/drive/MyDrive/Colab Notebooks/data/keypoint_cusntom2.hdf5")

import cv2



cap = cv2.imread(r"/content/drive/MyDrive/Colab Notebooks/data/152_284.jpg")
cap = cv2.resize(cap, (224,224))
imz= np.asarray(cap, dtype = 'float32')
norm_imz = imz/255.0 - 0.5 
scaler = preprocessing.MinMaxScaler (feature_range = (-0.5, 0.5))
datz = modelx.predict(norm_imz[None, :])

kp_unscale = scaler.inverse_transform(datz)
scaler = preprocessing.StandardScaler().fit(datz)
x = scaler.transform(x)

plt.imshow(norm_imz+0.5)
plt.plot(x[0][0], x[0][1],  marker='v', c='white')

import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from scipy.io import loadmat
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# loading the dataset
Train = loadmat('train_32x32.mat')
Test = loadmat('test_32x32.mat')
X_train = Train['X']
y_train = Train['y']
X_test = Test['X']
y_test = Test['y']
# converting the data to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = np.rollaxis(X_train, 3)
X_test = np.rollaxis(X_test, 3)
y_train = y_train[:, 0]
y_test = y_test[:, 0]
np.place(y_train, y_train == 10, 0)
np.place(y_test, y_test == 10, 0)
# normalizing the data
X_train /= 255
X_test /= 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()
# fully connected layer
model.build(input_shape=(32, 32, 3))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))
# getting a summery of built network

model.summary()
# adding the optimizer
adam = Adam(lr=1e-4, decay=1e-4 / 2)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# training the network
model.fit(X_train, y_train, batch_size=32, verbose=1, epochs=2, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('loss:', score[0])
print('Test accuracy:', score[1])

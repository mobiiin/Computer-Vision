import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.datasets import mnist
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
print("Shape of X_train is now:", X_train.shape)
print("Shape of X_test is now:", X_test.shape)
print("Shape of y_train is now:", y_train.shape)
print("Shape of y_test is now:", y_test.shape)

model = Sequential()
model.add(Convolution2D(32, (5, 5), padding='same', input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(64, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# converting the data into 1-dimensional array
model.add(Flatten())
# adding a classifier with 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))
# getting a summery of built network
model.summary()

# adding the optimizer
adam = Adam(lr=1e-4, decay=1e-4 / 2)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# training the network
model.fit(X_train, y_train, batch_size=32, verbose=2, epochs=10, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('loss:', score[0])
print('Test accuracy:', score[1])

# extracting the layers
layers = [model.get_layer('conv2d_1'),
          model.get_layer('conv2d_2'),
          model.get_layer('conv2d_3'),
          model.get_layer('conv2d_4')]

# Define a model which gives the outputs of the layers
layer_outputs = [layer.output for layer in layers]
activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
# Create a list with the names of the layers
layer_names = []
for layer in layers:
    layer_names.append(layer.name)


# Define a function which will plot the convolutional filters
def plot_convolutional_filters(img):
    img = np.expand_dims(img, axis=0)
    activations = activation_model.predict(img)
    images_per_row = 9

    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                :, :,
                                col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,
                row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='plasma')


img = X_train[42500]
plt.imshow(img)
plt.show()
plot_convolutional_filters(img)

### last section
im1 = cv.imread('858.png')



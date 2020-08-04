## acc : 0.982  이상 띄우기

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train[0])
print('y_train :',y_train[0])

## OneHotEncoding
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

## 정규화
# plt.imshow(x_train[0], 'gray')
# plt.show()
x_train = x_train.reshape(60000,784).astype('float32') / 255
x_test = x_test.reshape(10000,784).astype('float32') / 255

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, UpSampling2D, Conv2DTranspose, Input

input1 = Input(784,)
encoded = Dense(32,activation='relu')(input1)
decoded = Dense(784,activation='sigmoid')(encoded)
model = Model(inputs=input1, outputs=decoded)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])

model.fit(x_train, x_train, batch_size = 500, epochs=100, validation_split=0.2)


x_pred = model.predict(x_test)*255

for i, asdf in enumerate(x_pred):
    plt.subplot(2,1,1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.subplot(2,1,2)
    plt.imshow(asdf.reshape(28,28))
    plt.show()
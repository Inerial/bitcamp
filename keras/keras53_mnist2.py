## acc : 0.982  이상 띄우기

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train[0])
print('y_train :',y_train[0])


#plt.imshow(x_train[1], 'gray')
#plt.show()

#from sklearn.preprocessing import OneHotEncoder
#hot = OneHotEncoder()

## OneHotEncoding
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

## 정규화
#from sklearn.preprocessing import MinMaxScaler
x_train = x_train.reshape(60000,28,28,1).astype('float32') / 255 ## float 안해줘도 되지 않나?
x_test = x_test.reshape(10000,28,28,1).astype('float32') / 255

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(3, (2,2),padding='same', input_shape=(28,28,1)))
model.add(Conv2D(3, (2,2),padding='same'))
# model.add(MaxPool2D((2,2)))
model.add(Conv2D(3, (2,2),padding='same'))
model.add(Conv2D(1, (2,2),padding='same'))

# model.add(Flatten())

# model.add(Dense(200, activation='elu'))
# model.add(Dense(200, activation='elu'))
# model.add(Dense(200, activation='elu'))
# model.add(Dense(200, activation='elu'))
# model.add(Dense(200, activation='elu'))
# model.add(Dense(200, activation='elu'))
# model.add(Dense(200, activation='elu'))
# model.add(Dense(200, activation='elu'))
# model.add(Dense(200, activation='elu'))
# model.add(Dense(200, activation='elu'))
# model.add(Dense(200, activation='elu'))
# model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='mae', metrics=['mae'])

model.fit(x_train, x_train, batch_size = 100, epochs=30, validation_split=0.1)

loss, acc = model.evaluate(x_test,y_test)
print('loss :',loss)
print('acc :',acc)
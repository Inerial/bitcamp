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
print(y_train.shape)
## 정규화
#from sklearn.preprocessing import MinMaxScaler
plt.imshow(x_train[0], 'gray')
plt.show()
x_train = x_train.reshape(60000,28,28,1).astype('float32') / 255
x_test = x_test.reshape(10000,28,28,1).astype('float32') / 255

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(32, (5,5),padding='same',activation='elu', input_shape=(28,28,1)))
model.add(Conv2D(32, (5,5),padding='same',activation='elu'))
model.add(Conv2D(32, (5,5),padding='same',activation='elu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (5,5),padding='same',activation='elu'))
model.add(Conv2D(64, (5,5),padding='same',activation='elu'))
model.add(Conv2D(64, (5,5),padding='same',activation='elu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (5,5),padding='same',activation='elu'))
model.add(Conv2D(128, (5,5),padding='same',activation='elu'))
model.add(Conv2D(128, (5,5),padding='same',activation='elu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(256, (5,5),padding='same',activation='elu'))
model.add(Conv2D(256, (5,5),padding='same',activation='elu'))
model.add(Conv2D(256, (5,5),padding='same',activation='elu'))

model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.fit(x_train, y_train, batch_size = 500, epochs=10, validation_split=0.2)


print(model.predict(x_test))
loss, acc = model.evaluate(x_test,y_test)
print('loss :',loss)
print('acc :',acc)
from keras.datasets import fashion_mnist
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten, Input, LSTM, MaxPool2D
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()

## OneHotEncoding
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

## 정규화
x_train = x_train.reshape(60000,28*28).astype('float32') / 255
x_test = x_test.reshape(10000,28*28).astype('float32') / 255

model = Sequential()

model.add(Dense(200, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.fit(x_train, y_train, batch_size = 500, epochs=60, validation_split=0.3)

loss, acc = model.evaluate(x_test,y_test)
print('loss :',loss)
print('acc :',acc)
""" 
loss : 0.35533749743700027
acc : 0.8827999830245972 """
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten, Input, LSTM
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train),(x_test,y_test) = cifar10.load_data()


## OneHotEncoding
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

## 정규화
x_train = x_train.reshape(50000,32*32*3).astype('float32') / 255
x_test = x_test.reshape(10000,32*32*3).astype('float32') / 255


input1 = Input(shape = (32*32*3, ))
Dense1 = Dense(1536, activation='elu')(input1)
Dense1 = Dropout(0.3)(Dense1)
Dense1 = Dense(768, activation='elu')(Dense1)
Dense1 = Dropout(0.3)(Dense1)
Dense1 = Dense(384, activation='elu')(Dense1)
Dense1 = Dropout(0.3)(Dense1)
Dense1 = Dense(192, activation='elu')(Dense1)
Dense1 = Dropout(0.3)(Dense1)
Dense1 = Dense(96, activation='elu')(Dense1)
Dense1 = Dropout(0.3)(Dense1)
Dense1 = Dense(48, activation='elu')(Dense1)
Dense1 = Dropout(0.3)(Dense1)
Dense1 = Dense(24, activation='elu')(Dense1)
Dense1 = Dropout(0.3)(Dense1)
Dense1 = Dense(10, activation='softmax')(Dense1)

model = Model(inputs = input1, outputs = Dense1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.fit(x_train, y_train, batch_size = 500, epochs=120, validation_split=0.3)

loss, acc = model.evaluate(x_test,y_test)
print('loss :',loss)
print('acc :',acc)
""" 
loss : 1.4997556520462036
acc : 0.5519999861717224 """
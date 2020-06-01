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
x_train = x_train.reshape(60000,28,28,1).astype('float32') / 255
x_test = x_test.reshape(10000,28,28,1).astype('float32') / 255




from keras.models import load_model
from keras.layers import Dense

model = load_model('./model/model_test01.h5')
model.add(Dense(100, name='1'))
model.add(Dense(100, name='2'))
model.add(Dense(10, name='3'))

model.summary()

loss, acc = model.evaluate(x_test,y_test)
print('loss :',loss)
print('acc :',acc)

## 이전
""" loss : 0.13090835977123425
acc : 0.9800999760627747 """
## 이후
""" loss : 8.489140173339845
acc : 0.18479999899864197 """
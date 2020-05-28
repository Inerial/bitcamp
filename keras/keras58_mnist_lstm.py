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

# time_steps, size = [784 , 1]
# time_steps, size = [392 , 2]
# time_steps, size = [196 , 4]
# time_steps, size = [112 , 7]
# time_steps, size = [98 , 8]
# time_steps, size = [56 , 14]
# time_steps, size = [49 , 16]

time_steps, size = [28 , 28] # 0.989

# time_steps, size = [16 , 49] ## 0.980
# time_steps, size = [14 , 56] ## 0.988
# time_steps, size = [8 , 98] ## 0.980
# time_steps, size = [7 , 112] ## 0.985
# time_steps, size = [4 , 196] ## 0.984
# time_steps, size = [2 , 392] ## 0.984
# time_steps, size = [1 , 784] ## 0.980

## 정규화
#from sklearn.preprocessing import MinMaxScaler
x_train = x_train.reshape(60000,time_steps,size).astype('float32') / 255 ## float 안해줘도 되지 않나?
x_test = x_test.reshape(10000,time_steps,size).astype('float32') / 255


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(800, activation ='elu', input_shape= (time_steps,size)))
model.add(Dropout(0.2))

model.add(Dense(100, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.fit(x_train, y_train, batch_size = 500, epochs=60, validation_split=0.3)

loss, acc = model.evaluate(x_test,y_test)
print('loss :',loss)
print('acc :',acc)

## dropout : 노드를 일정 비율 비활성화시키면서 노드를 적합
#0.989
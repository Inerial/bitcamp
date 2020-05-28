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

# time_steps, size = [784 , 1] ## 에러
# time_steps, size = [392 , 2] ## 에러
# time_steps, size = [196 , 4] ## 에러
# time_steps, size = [112 , 7] ## 0.487
# time_steps, size = [98 , 8] ## 0.118
# time_steps, size = [56 , 14] ## 0.988
# time_steps, size = [49 , 16] ## 0.981

# time_steps, size = [28 , 28] # 0.989

# time_steps, size = [16 , 49] ## 0.980
# time_steps, size = [14 , 56] ## 0.988
# time_steps, size = [8 , 98] ## 0.980
# time_steps, size = [7 , 112] ## 0.985
# time_steps, size = [4 , 196] ## 0.984
# time_steps, size = [2 , 392] ## 0.984
time_steps, size = [1 , 784] ## 0.982

## time_step이 증가할수록 실행 속도가 느려진다 (증가폭에 비례하는거보다 더 증가하는것으로 보임).
## 검정력의 차이는 딱히 보이지 않는다 오히려 타임스텝이 너무 커지면 검정력이 낮아짐
## time_step이 특정값을 넘어가면 에러가 뜬다. (메모리를 겁나게 잡아먹는것으로 보임)
## 따라서 시계열적 특성이 없는 데이터라면 time_step을 줄 이유가 없다. 라고 볼 수 있다.


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
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

#1개짜리 데이터를 넣을떄 input_dim = 1
x = x.reshape(x.shape[0], x.shape[1], 1)

#2. 모델구성
Input1 = Input(shape = (3,1))
#model.add(LSTM(10, activation='relu', input_shape=(3,1))) 
lstm1 = LSTM(100, activation='relu')(Input1) 
dense1 = Dense(100)(lstm1)
dense1 = Dense(100)(dense1)
dense1 = Dense(1)(dense1)

model = Model(inputs=Input1, outputs=dense1)

#3. 실행
model.compile(loss = 'mse', optimizer='adam')
model.fit(x,y,epochs=7000,batch_size =32)

x_pred = np.array([5,6,7])
x_pred = x_pred.reshape(1,3,1)  ## 같은 크기의 행렬로 만들어줌

y_pred = model.predict(x_pred)
print(y_pred)
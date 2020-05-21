import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM

#1. 데이터
x1 = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
x2 = np.array([[10,20,30], [20,30,40], [30,40,50], [40,50,60],[50,60,70],[60,70,80],[70,80,90],[80,90,100],[90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_pred = np.array([55,65,75])
x2_pred = np.array([65,75,85])

#1개짜리 데이터를 넣을떄 input_dim = 1
x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)


#2. 모델구성
input1 = Input(shape=(3,1))
lstm1 = LSTM(100, activation = 'relu')(input1) 
dense1 = Dense(100)(lstm1)
dense1 = Dense(100)(dense1)
dense1 = Dense(100)(dense1)



input2 = Input(shape=(3,1))
lstm2 = LSTM(100, activation = 'relu')(input2) 
dense2 = Dense(100)(lstm2)
dense2 = Dense(100)(dense2)
dense2 = Dense(100)(dense2)

from keras.layers import concatenate
middle1 = concatenate(inputs = [dense1, dense2])


middle1 = Dense(1)(middle1)

# 데이터의 개수인 행은 무시하고 x의 shape
model = Model(inputs=[input1,input2], outputs=middle1)
model.summary()

#3. 실행
model.compile(loss = 'mae', optimizer='adam')
model.fit([x1,x2],y,epochs=2000,batch_size = 32)

x1_pred = x1_pred.reshape(1,3,1)  ## 같은 크기의 행렬로 만들어줌
x2_pred = x2_pred.reshape(1,3,1)  ## 같은 크기의 행렬로 만들어줌

y_pred = model.predict([x1_pred, x2_pred])
print(y_pred)
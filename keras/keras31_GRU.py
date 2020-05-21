import numpy as np
from keras.models import Sequential
from keras.layers import Dense, GRU

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape) # (4,3)
print(y.shape) # (4,)
#1개짜리 데이터를 넣을떄 input_dim = 1
x = x.reshape(4,3,1)
#x = x.reshape(x.shape[0], x.shape[1], 1)
'''
                행          열       자르는 개수
x의 shape = (batch_size, time_steps, feature)
input_shape = (time_steps, feature)
input_length = time_steps, input_dim = feature
'''
# 4행 3열짜리 데이터를 한개씩 꺼내쓰겠다는 뜻

#2. 모델구성
model = Sequential()
#model.add(LSTM(10, activation='relu', input_shape=(3,1))) 
model.add(GRU(100, input_length = 3, input_dim = 1)) 
# 데이터의 개수인 행은 무시하고 x의 shape
model.add(Dense(100))
model.add(Dense(1))

model.summary()

#3. 실행
model.compile(loss = 'mse', optimizer='adam')
model.fit(x,y,epochs=7000,batch_size =32)

x_pred = np.array([5,6,7])
x_pred = x_pred.reshape(1,3,1)  ## 같은 크기의 행렬로 만들어줌

y_pred = model.predict(x_pred)
print(y_pred)
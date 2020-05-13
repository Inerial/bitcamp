#1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense #DNN구조의 베이스가 되는 구조
model = Sequential()

model.add(Dense(5,input_dim = 1))
model.add(Dense(3))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
##cpu에서 두개이상 쓰면 에러가 난다
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics = ['acc'])
model.fit(x,y, epochs=30, batch_size=1)

#4. 평가와 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print("loss : " , loss , '\n' , "acc : " , acc) 
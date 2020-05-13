#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16,17,18])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense #DNN구조의 베이스가 되는 구조
model = Sequential()

model.add(Dense(20,input_dim = 1))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))

## 두가지 방법 회귀와 분류

#3. 훈련
## MSE는 mean square error로 예측한 값과 실제 값의 차이(잔차)의 제곱 평균을 말한다. == 회귀지표
## acc는 분류지표 == 서로 다름
model.compile(loss='mse', optimizer='adam', metrics = ['mse'])
model.fit(x_train ,y_train , epochs=100, batch_size=1)

#4. 평가와 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("loss : " , loss , '\n' , "mse : " , mse)

y_pred = model.predict(x_pred)
print("y_predict : ", y_pred)
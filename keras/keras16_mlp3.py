#1. 데이터
## transpose해서 행렬을 바꿔준다. (행 : 데이터개수, 열 : 각 데이터가 가지고있는 값들)
import numpy as np
x = np.array([range(1,101)]).T
y = np.array([range(101,201), range(711,811), range(100)]).T

from sklearn.model_selection import train_test_split
## train, test값을 train 0.8사이즈로 분류
x_train, x_test, y_train, y_test = train_test_split(
    x,y ,random_state = 66,train_size = 0.8
)
#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense #DNN구조의 베이스가 되는 구조

# 데이터가 복잡하지 않기 때문에 모델을 그렇게 크게 안잡는게 더 좋았다.
## 바꾼 데이터의 input , output되는 데이터의 개수가 3, 3개이므로 첫 input과 마지막 output을 3으로 잡아주어야한다.
model = Sequential()
model.add(Dense(40,input_dim = 1))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(3))


#3. 훈련
## MSE는 mean square error로 예측한 값과 실제 값의 차이(잔차)의 제곱 평균을 말한다. == 회귀지표
## acc는 분류지표 == 서로 다름
model.compile(loss='mse', optimizer='adam', metrics = ['mse'])
## 받은 train 데이터에서 0.25크기만큼 validation 데이터로 활용
## train 데이터는 전체 데이터의 0.8크기이므로 val데이터는 전체의 0.2
model.fit(x_train ,y_train , epochs=200, batch_size=3, validation_split=0.25)


#4. 평가와 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("loss : " , loss , '\n' , "mse : " , mse)


##y_pred = model.predict(x_pred)
##print("y_predict : ", y_pred)

## x_test값을 이용하여 y_test의 추정치 생산
y_pred = model.predict(x_test)
print(y_pred)

## sklearn의 mse를 이용하여 rmse함수 생성
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
##rmse함수 실행
print("RMSE : ", RMSE(y_test, y_pred))


#회귀모델의 지표
#mse : mean squared error
#rmse : mse의 제곱근
#이 둘의 단점? 이 둘은 어디쯤이 최선인지 확실하지 않다
#이때 사용하는것이 결정계수 R2
#mean value로 예측하는 단순모델과 비교하여 상대적인 성능을 측정한다
# R^2 = 1 - 오차제곱합/편차제곱합

# 오차제곱합이란? SE == 평균을 안낸 mse
# 편차제곱합이란? ST == 실제값과 실제값의 평균값간의 차이 == 분산 * n

## 결정계수 구하기
from sklearn.metrics import r2_score
r2_y = r2_score(y_test,y_pred)
print("결정계수 : ", r2_y)

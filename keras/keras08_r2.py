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

model.add(Dense(40,input_dim = 1))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(1))

## 두가지 방법 회귀와 분류

## 학생수능점수, 온도, 날씨, 하이닉스, 유가, 환율, 금시계, 금리 등으로 삼성주가등을 사용 가능 (피쳐 임포턴스)
## 피처 임포턴스 위의 각각 변수
## train, test를 한 데이터에서 %로 나누어서 각각 진행
##다양항 변수를 고려해줘야한다.
    

#3. 훈련
## MSE는 mean square error로 예측한 값과 실제 값의 차이(잔차)의 제곱 평균을 말한다. == 회귀지표
## acc는 분류지표 == 서로 다름
model.compile(loss='mse', optimizer='adam', metrics = ['mse'])
model.fit(x_train ,y_train , epochs=100, batch_size=1)

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

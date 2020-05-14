## 과제 : R2 가 음수가 아닌 0.5 이하로 줄이기.
## 레이어는 인풋과 아웃풋을 포함 5개 이상, 노드는 레이어당 각각 5개 이상
## batch_size = 1, epochs = 100 이상

#1. 데이터
import numpy as np
x = np.array(range(1,101))
y = np.array(range(101,201))
x_train = x[:60]
y_train = y[:60]
x_val = x[60:80]
y_val = y[60:80]
x_test = x[80:]
y_test = y[80:]

#x_pred = np.array([16,17,18])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense #DNN구조의 베이스가 되는 구조
from keras.activations import sigmoid


##과제의 조건을 맞추기위한 모델링방법
##결정계수의 식은 1 - 잔차제곱합/편차제곱합
## 이는 1 - 시그마(실제값 - 예측값)/시그마(실제값 - 평균) < 0.5 가 되어야 한다.
## 이때 예측값이 모두 평균에 수렴하는 값이 나올수록 결정계수의 값이 작아진다고 생각했다.
## 따라서  y = wx + b 의 형태를 가진 식에서 weight가 0에 가까워지고, b가 평균에 가까워지면 결정계수의 값이 0에 가까워질것이라고 판단했다.
## 검색을 통해 vanishing gradient problem을 알게되었고 이때 sigmoid function을 사용하면 weight가 제데로 적용이 안되는 문제가 발생한다는것을 확인하였다.
## 따라서 모델의 윗부분에 sigmoid 함수를 적용하고, 아랫부분은 그대로 두어 w의 값은 제데로 적용이 안되면서 mse를 줄이려는 움직임이
## 결정계수의 값을 0에 수렴하게 만들지 않을까 생각했고 이 생각은 어느정도는 맞았다.
## 하지만 편차가 정확하게 평균가까이 안착하지 못하는 경우가 많아 결정계수가 음수로 가는 경우의 수가 많이 발견되었다.(절댓값으로 치면 거의다 0.5이하)
model = Sequential()

model.add(Dense(5,input_dim = 1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

## 두가지 방법 회귀와 분류

## 학생수능점수, 온도, 날씨, 하이닉스, 유가, 환율, 금시계, 금리 등으로 삼성주가등을 사용 가능 (피쳐 임포턴스)
## 피처 임포턴스 위의 각각 변수
## train, test를 한 데이터에서 %로 나누어서 각각 진행
##다양항 변수를 고려해줘야한다.

##18000번 가량

#3. 훈련
## MSE는 mean square error로 예측한 값과 실제 값의 차이(잔차)의 제곱 평균을 말한다. == 회귀지표
## acc는 분류지표 == 서로 다름
model.compile(loss='mse', optimizer='adam', metrics = ['mse'])
model.fit(x_train ,y_train , epochs=200, batch_size=2, validation_data=(x_val, y_val))


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

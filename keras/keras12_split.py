
#1. 데이터
import numpy as np
x = np.array(range(1,101))
y = np.array(range(101,201))
'''
x_train = x[:60]
y_train = y[:60]
x_val = x[60:80]
y_val = y[60:80]
x_test = x[80:]
y_test = y[80:]
'''
from sklearn.model_selection import train_test_split

##1번
x_train, x_test, y_train, y_test = train_test_split(
    #x,y ,random_state = 6666,train_size = 0.6
    x,y, shuffle = False, train_size = 0.7
)
x_test, x_val, y_test, y_val = train_test_split(
    #x_test,y_test,shuffle = False, random_state = 6666, test_size = 0.5
    x_test, y_test, shuffle = False, test_size = 12 / 30
 )

'''
##2번
x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state = 66, train_size=0.6, test_size = 0.2,
 )

 ##중복되는값이 있으면 하나만 들어가도 죄다 빼버리므로 안됨
x_val = np.setdiff1d(np.setdiff1d(x, x_train), x_test)
y_val = np.setdiff1d(np.setdiff1d(y, y_train), y_test)
#x_pred = np.array([16,17,18])
'''

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense #DNN구조의 베이스가 되는 구조



model = Sequential()
model.add(Dense(15,input_dim = 1))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(15))
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
model.fit(x_train ,y_train , epochs=100, batch_size=3, validation_data=(x_val, y_val))


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

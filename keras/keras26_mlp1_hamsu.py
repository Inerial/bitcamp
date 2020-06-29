#1. 데이터
import numpy as np
x = np.array([range(1,101),range(311,411),range(100)]).T
y = np.array([range(101,201), range(711,811), range(100)]).T

from sklearn.model_selection import train_test_split
## train, test값을 train 0.8사이즈로 분류
x_train, x_test, y_train, y_test = train_test_split(
    x,y ,random_state = 66,train_size = 0.8
)
#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(3,))
dense1 = Dense(50)(input1)
dense1 = Dense(50)(dense1)
dense1 = Dense(3)(dense1)

model = Model(inputs=input1, outputs=dense1)


model.summary()

model.compile(loss='mse', optimizer='adam', metrics = ['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience=10, mode = 'auto')
model.fit(x_train ,y_train , epochs=1000, batch_size=3, validation_split=0.25, callbacks=[early_stopping])


#4. 평가와 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=3)
print("loss : " , loss , '\n' , "mse : " , mse)


## x_test값을 이용하여 y_test의 추정치 생산
y_pred = model.predict(x_test)
print(y_pred)

## sklearn의 mse를 이용하여 rmse함수 생성
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
##rmse함수 실행
print("RMSE : ", RMSE(y_test, y_pred))


## 결정계수 구하기
from sklearn.metrics import r2_score
r2_y = r2_score(y_test,y_pred)
print("결정계수 : ", r2_y)

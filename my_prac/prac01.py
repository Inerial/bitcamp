#이차함수를 전처리하지 않고 회귀식으로 딥러닝하는것은 힘들어보인다.
#그래도 이것저것 해보자

#1. 데이터
import numpy as np
x1 = np.array([range(1,101)]).T
x2 = np.array([range(1,101)]).T
y = x1*x1 + x2 + 200

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1,x2,y ,train_size = 0.8
)
#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(1,))
dense1 = Dense(50)(input1)
dense1 = Dense(50)(dense1)
dense1 = Dense(50)(dense1)
dense1 = Dense(50)(dense1)
dense1 = Dense(50)(dense1)
dense1 = Dense(50)(dense1)
dense1 = Dense(50)(dense1)
dense1 = Dense(50)(dense1)
dense1 = Dense(50)(dense1)
dense1 = Dense(50)(dense1)
dense1 = Dense(50)(dense1)
dense1 = Dense(50)(dense1)
dense1 = Dense(50)(dense1)
dense1 = Dense(50)(dense1)
dense1 = Dense(50)(dense1)
dense1 = Dense(50)(dense1)
dense1 = Dense(50)(dense1)

input2 = Input(shape=(1,))
dense2 = Dense(50)(input2)
dense2 = Dense(50)(dense2)
dense2 = Dense(50)(dense2)

from keras.layers import concatenate
middle = concatenate([dense1, dense2])
middle = Dense(50)(middle)
middle = Dense(50)(middle)
middle = Dense(1)(middle)

model = Model(inputs=[input1, input2], outputs=middle)


model.compile(loss='mse', optimizer='adam', metrics = ['mse'])

model.summary()

from keras.callbacks import EarlyStopping
#early_stopping = EarlyStopping(monitor = 'loss', patience=10, mode='auto')
model.fit([x1_train,x2_train] ,y_train , epochs=1000, batch_size=3, validation_split=0.25)#, callbacks=[early_stopping])


#4. 평가와 예측
loss, mse = model.evaluate([x_test, x2_test], y_test, batch_size=3)
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

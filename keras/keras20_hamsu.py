

import numpy as np
x = np.array([range(1,101),range(311,411),range(100)]).T
y = np.array([range(711,811)]).T

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y ,random_state = 66,train_size = 0.8
)

from keras.models import Sequential, Model
from keras.layers import Dense
'''
model = Sequential()
model.add(Dense(5,input_dim = 3))
model.add(Dense(4))
model.add(Dense(1))
'''
Input



model.compile(loss='mse', optimizer='adam', metrics = ['mse'])

## verbose = 0 : 과정을 보여주지 않는다.
## verbose = 1 : 디폴트
## verbose = 2 : 진행과정이 안나옴
## verbose = 3 : epochs위치만 확인
model.fit(x_train ,y_train , epochs=100, batch_size=3, validation_split=0.25, verbose=0)

loss, mse = model.evaluate(x_test, y_test, batch_size=3)
print("loss : " , loss , '\n' , "mse : " , mse)


y_pred = model.predict(x_test)
print(y_pred)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE : ", RMSE(y_test, y_pred))


from sklearn.metrics import r2_score
r2_y = r2_score(y_test,y_pred)
print("결정계수 : ", r2_y)

#1. R2 0.5 이하
#2. layers 5개 이상
#3. Nodes 10개 이상
#4. batch_size 8 이하
#5. epochs 100이상

import numpy as np
x = np.array([range(1,101),range(311,411),range(100)]).T
y = np.array([range(711,811)]).T

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,y ,random_state = 66,train_size = 0.8
)
from keras.models import Sequential
from keras.layers import Dense 

model = Sequential()
model.add(Dense(10,input_dim = 3))
model.add(Dense(10))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics = ['mse'])
model.fit(x_train ,y_train , epochs=1000, batch_size=8, validation_split=0.25)

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


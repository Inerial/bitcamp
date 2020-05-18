

import numpy as np
x = np.array([range(1,101),range(311,411),range(100)]).T
y = np.array([range(711,811)]).T

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y ,random_state = 66,train_size = 0.8
)

from keras.models import Sequential, Model
from keras.layers import Dense, Input
'''
model = Sequential()
model.add(Dense(5,input_dim = 3))
model.add(Dense(4))
model.add(Dense(1))
'''
input1 = Input(shape = (3,))

dense1 = Dense(25, activation='relu')(input1)
dense2 = Dense(15, activation='relu')(dense1)
dense3 = Dense(5, activation='relu')(dense2)
dense4 = Dense(15, activation='relu')(dense3)
dense6 = Dense(25, activation='relu')(dense4)

output1 = Dense(1)(dense6)


model = Model(inputs=input1, outputs = output1)
model.summary()

model.compile(loss='mse', optimizer='adam', metrics = ['mse'])


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

# 1. 데이터
import numpy as np

x1 = np.array([range(1,101),range(311,411), range(411,511)]).T
x2 = np.array([range(711,811), range(711,811), range(511,611)]).T

y1 = np.array([range(101,201),range(411,511), range(100)]).T

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1,y1 ,random_state = 66,train_size = 0.8
)

x2_train, x2_test = train_test_split(
    x2,random_state = 66,train_size = 0.8
)

##2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape = (3,))
dense1_1 = Dense(25, activation='relu', name="left-input1")(input1)
dense1_2 = Dense(15, activation='relu', name="left-input2")(dense1_1)
dense1_3 = Dense(5, activation='relu', name="left-input3")(dense1_2)
dense1_4 = Dense(15, activation='relu', name="left-input4")(dense1_3)
dense1_5 = Dense(20, activation='relu', name="left-input5")(dense1_4)

input2 = Input(shape = (3,))
dense2_1 = Dense(25, activation='relu', name="right-input1")(input2)
dense2_2 = Dense(15, activation='relu', name="right-input2")(dense2_1)
dense2_3 = Dense(5, activation='relu', name="right-input3")(dense2_2)
dense2_4 = Dense(5, activation='relu', name="right-input4")(dense2_3)

from keras.layers.merge import concatenate
merge1 = concatenate([dense1_5, dense2_4])

###### output

middle1 = Dense(30, name="middle1")(merge1)
middle1 = Dense(5, name="middle2")(middle1)
middle1 = Dense(7, name="middle3")(middle1)


output1_1 = Dense(30, name="left-output1")(middle1)
output1_2 = Dense(7, name="left-output2")(output1_1)
output1_3 = Dense(3, name="left-output3")(output1_2)

model = Model(inputs = [input1, input2], outputs=[output1_3])
model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics = ['mse'])
model.fit([x1_train, x2_train] ,[y1_train] , epochs=100, batch_size=3, validation_split=0.25, verbose=1)

eval = model.evaluate([x1_test, x2_test], [y1_test], batch_size=1)
if(len(eval) != 2):
    print("loss_total : " , eval[0])
    esize = np.int(np.size(eval)/2)
    for i in range(esize):
        print("loss",i+1,": ", eval[i+1])
        print("mse",i+1,": ", eval[i+1 + esize])
else:
    print("loss : " , eval[0], "\nmse : " , eval[1])
y1_pred = model.predict([x1_test,x2_test])
print(y1_pred)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
RMSE1 = RMSE(y1_test, y1_pred)

print("RMSE1 : ", RMSE1)


from sklearn.metrics import r2_score
r2_y1 = r2_score(y1_test,y1_pred)
print("y1결정계수 : ", r2_y1)
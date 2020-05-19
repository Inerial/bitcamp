# 1. 데이터
import numpy as np

x1 = np.array([range(1,101), range(301,401)]).T

y1 = np.array([range(711,811), range(611,711)]).T
y2 = np.array([range(101,201), range(411,511)]).T

from sklearn.model_selection import train_test_split
x1_train, x1_test,y1_train,y1_test, y2_train, y2_test = train_test_split(
    x1,y1,y2 ,random_state = 66,train_size = 0.8
)

##2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape = (2,))
'''
dense1 = Dense(25, activation='relu', name="left-input1")(input1)
dense1 = Dense(15, activation='relu', name="left-input2")(dense1)
###### output

middle1 = Dense(10, name="middle1")(dense1)
'''
dense2 = Dense(300)(input1)
dense2 = Dense(120)(dense2)
dense2 = Dense(20)(dense2)
dense2 = Dense(20)(dense2)
dense2 = Dense(2)(dense2)

dense3 = Dense(300)(input1)
dense3 = Dense(120)(dense3)
dense3 = Dense(20)(dense3)
dense3 = Dense(20)(dense3)
dense3 = Dense(2)(dense3)

model = Model(inputs = [input1], outputs=[dense2, dense3])
model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics = ['mse'])
model.fit([x1_train] ,[y1_train, y2_train] , epochs=100, batch_size=3, validation_split=0.25, verbose=1)

eval = model.evaluate([x1_test], [y1_test, y2_test], batch_size=1)
if(len(eval) != 2):
    print("loss_total : " , eval[0])
    esize = np.int(np.size(eval)/2)
    for i in range(esize):
        print("loss",i+1,": ", eval[i+1])
        print("mse",i+1,": ", eval[i+1 + esize])
else:
    print("loss : " , eval[0], "\nmse : " , eval[1])


y1_pred, y2_pred = model.predict([x1_test])
print(y1_pred)
print(y2_pred)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
RMSE1 = RMSE(y1_test, y1_pred)
RMSE2 = RMSE(y2_test, y2_pred)

print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE_mean : ", (RMSE1+RMSE2)/2)


from sklearn.metrics import r2_score
r2_y1 = r2_score(y1_test,y1_pred)
r2_y2 = r2_score(y2_test,y2_pred)

print("결정계수 : ", r2_y1)
print("결정계수 : ", r2_y2)
print("결정계수 : ", (r2_y1+r2_y2)/2)
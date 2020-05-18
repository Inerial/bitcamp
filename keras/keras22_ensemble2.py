# 1. 데이터
import numpy as np

x1 = np.array([range(1,101),range(311,411)]).T
x2 = np.array([range(711,811), range(711,811)]).T

y1 = np.array([range(101,201),range(411,511)]).T
y2 = np.array([range(501,601), range(711,811)]).T
y3 = np.array([range(411,511), range(611,711)]).T

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1,y1 ,random_state = 66,train_size = 0.8
)

x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2,y2 ,random_state = 66,train_size = 0.8
)

y3_train, y3_test = train_test_split(
    y3 ,random_state = 66,train_size = 0.8
)

##2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape = (2,))
dense1_1 = Dense(25, activation='relu', name="left-input1")(input1)
dense1_2 = Dense(15, activation='relu', name="left-input2")(dense1_1)
dense1_3 = Dense(5, activation='relu', name="left-input3")(dense1_2)
dense1_4 = Dense(15, activation='relu', name="left-input4")(dense1_3)
dense1_5 = Dense(20, activation='relu', name="left-input5")(dense1_4)

input2 = Input(shape = (2,))
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
output1_3 = Dense(2, name="left-output3")(output1_2)

output2_1 = Dense(30, name="middle-output1")(middle1)
output2_2 = Dense(7, name="middle-output2")(output2_1)
output2_3 = Dense(2, name="middle-output3")(output2_2)

output3_1 = Dense(30, name="right-output1")(middle1)
output3_2 = Dense(7, name="right-output2")(output3_1)
output3_3 = Dense(2, name="right-output3")(output3_2)

model = Model(inputs = [input1, input2], outputs=[output1_3,output2_3, output3_3])
model.summary()


model.compile(loss='mse', optimizer='adam', metrics = ['mse'])


model.fit([x1_train, x2_train] ,[y1_train, y2_train, y3_train] , epochs=100, batch_size=3, validation_split=0.25)

eval = model.evaluate([x1_test, x2_test], [y1_test,y2_test,y3_test], batch_size=1)

##asdf = dict(zip(["loss_total", "loss1","loss2","mse1","mse2"], model.evaluate([x1_test, x2_test], [y1_test,y2_test], batch_size=1)))

print("loss_total : " , eval[0])
esize = np.int(np.size(eval)/2)
for i in range(esize):
    print("loss",i+1,": ", eval[i+1])
    print("mse",i+1,": ", eval[i+1 + esize])

y1_pred, y2_pred, y3_pred = model.predict([x1_test,x2_test])
print(y1_pred)
print(y2_pred)
print(y3_pred)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
RMSE1 = RMSE(y1_test, y1_pred)
RMSE2 = RMSE(y2_test, y2_pred)
RMSE3 = RMSE(y3_test, y3_pred)


print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE3 : ", RMSE3)
print("RMSE : ", (RMSE1+RMSE2+RMSE3)/3)
print("RMSE_total : ", RMSE(np.vstack([y1_test, y2_test, y3_test]), np.vstack([y1_pred, y2_pred, y3_pred])))


from sklearn.metrics import r2_score
r2_y1 = r2_score(y1_test,y1_pred)
r2_y2 = r2_score(y2_test,y2_pred)
r2_y3 = r2_score(y3_test,y3_pred)
r2_total = r2_score(np.vstack([y1_test, y2_test, y3_test]), np.vstack([y1_pred, y2_pred, y3_pred]))
print("y1결정계수 : ", r2_y1)
print("y2결정계수 : ", r2_y2)
print("y3결정계수 : ", r2_y3)
print("결정계수 : ", (r2_y1 + r2_y2 + r2_y3)/3)
print("total 결정계수 : ", r2_total)

def rMSE(y_test, y_pred):
    if isinstance(y_test,list):
        answer = []
        for i in range(len(y_test)):
            answer.append(np.sqrt(mean_squared_error(y_test[i], y_pred[i])))
        return answer
    else :
        return np.sqrt(mean_squared_error(y_test, y_pred))
    
print(rMSE([y1_test, y2_test, y3_test],[y1_pred, y2_pred, y3_pred]))
print(rMSE(y1_test, y1_pred))
print(RMSE(y1_test,y1_pred))
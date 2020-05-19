# 1. 데이터
import numpy as np

x1 = np.array([range(1,101),range(311,411), range(411,511)]).T
x2 = np.array([range(711,811), range(711,811), range(511,611)]).T

y1 = np.array([range(101,201),range(411,511), range(100)]).T

from sklearn.model_selection import train_test_split
x1_train, x1_test,x2_train,x2_test, y1_train, y1_test = train_test_split(
    x1,x2,y1 ,random_state = 66,train_size = 0.8
)

##2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape = (3,))
dense1 = Dense(50, activation='relu')(input1)
dense1 = Dense(50, activation='relu')(dense1)

input2 = Input(shape = (3,))
dense2 = Dense(50, activation='relu')(input2)
dense2 = Dense(50, activation='relu')(dense2)

from keras.layers.merge import concatenate
merge1 = concatenate([dense1, dense2])

###### output

middle1 = Dense(3, name="middle1")(merge1)


model = Model(inputs = [input1, input2], outputs=[middle1])
model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics = ['mse'])

from keras.callbacks import EarlyStopping
# mode는 min, max, auto가 있으며 min은 최솟값에서, max는 최댓값에서, auto는 둘중하나 탐지해서 각 값에서 10번흔들렸을때 값을 구한다
early_stopping = EarlyStopping(monitor="loss", patience=10,mode = "auto")
model.fit([x1_train, x2_train] ,[y1_train] , epochs=1000, batch_size=3, validation_split=0.25, verbose=1, callbacks=[early_stopping])

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

print("RMSE : ", RMSE1)


from sklearn.metrics import r2_score
r2_y1 = r2_score(y1_test,y1_pred)
print("결정계수 : ", r2_y1)

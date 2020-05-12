from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense,Dropout,Activation,Embedding,LSTM
import numpy as np

##절편의 값이 커지면 정확도가 떨어진다?
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([109,117,125,133,141,149,157,165,173,181])
x_test = np.array([100,101,102,103,104])
y_test = np.array([901,909,917,925,933])


## 노드, 레이어의 개수를 정하는 근거는? 쌓은 경험과 외부의 좋은 모델을 가져와서 해결하는것이 가장 가성비가 좋다.
## output, input 순으로 노드 개수를 정함, add한만큼 레이어 개수
## bias의 값이 큰것으로 보이면 레이어의 개수를 늘려야 적합이 잘 되는 것으로 보인다.
## 반대로 weight의 값이 크면 노드의 개수를 늘려야 적합이 잘 되는것으로 보인다.
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1, activation='relu'))

##모델의 파라미터 개수(한 시행동안 연산 개수)를 알수 있다.
model.summary()

model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])

##적합과정,  epochs = 시행횟수, batch_size = 한번에  x를 쓰는 횟수
##batch_size 디폴트값 32 (train의 개수를 100개로 두면 32개씩 처리하는것을 확인할 수 있었다.)
model.fit(x_train, y_train, epochs=500, batch_size = 1, validation_data = (x_test, y_test))
loss,acc = model.evaluate(x_test ,y_test, batch_size=1 )

print("loss : ", loss)
print("acc : ",acc)

output = model.predict(x_test)
print("결과물 : \n", output)

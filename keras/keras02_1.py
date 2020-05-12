from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense,Dropout,Activation,Embedding,LSTM
import numpy as np

##절편의 값이 커지면 정확도가 떨어진다?
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([201,504,1009,1716,2625,3736,5049,6564,8281,10200])
x_train = x_train[0:9]
y_train = y_train[1:10] - y_train[0:9]

x_test = np.array([11,12,13,14,15])
y_test = np.array([12321,14644,17169,19896,22825])
x_test = x_test[0:4]
y_test = y_test[1:5] - y_test[0:4]
## 2차함수의 형태는 불가능 == 무조건 모델의 결과는 y = wx + b 의 형태로 예측됨 ==> y의 값이 선형의 형태로 보이지 않다면 가공을 통해 선형으로 바꿔줄 필요성이 있다.

## 만약 미분을 통해 가공한다면, y = wx + b 의 형태로 계산된 예측값을 어떻게 실제 예측값으로 변경하는가? (x를 넣었을때 y의 미분값만 덜렁 튀어나오면 실제값을 어케암? == 함수를 알아야됨)
## 데이터의 시각화를 통해 대략적인 그래프의 형태를 맞추고 각각의 계수를 알아둘 생각을 해야한다. (사실 미분 비스무리한 방식으로 함수 안구해도 뒷수를 앞수로 빼주기만 해도 대충 된다.)

## 노드, 레이어의 개수를 정하는 근거는? 쌓은 경험과 외부의 좋은 모델을 가져와서 해결하는것이 가장 가성비가 좋다.
## output, input 순으로 노드 개수를 정함, add한만큼 레이어 개수
## bias나 weight의 값이 큰것으로 보이면 레이어의 개수를 늘려야 적합이 잘 되는 것으로 보인다. (둘 다 작을때는 상관없음)
## weight는 노드의 개수를 늘려도 적합이 잘 되는것으로 보인다.
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
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1, activation='relu'))

##모델의 파라미터 개수(한 시행동안 연산 개수)를 알수 있다.
##bias와 weight 값을 추정하고, 적당한 비율을 맞추어 연산속도 또한 맞춰줄 필요가 있다.
model.summary()

model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])

##적합과정,  epochs = 시행횟수, batch_size = 한번에  x를 쓰는 횟수
##batch_size 디폴트값 32 (train의 개수를 100개로 두면 32개씩 처리하는것을 확인할 수 있었다.)
model.fit(x_train, y_train, epochs=200, batch_size = 1, validation_data = (x_test, y_test))
loss,acc = model.evaluate(x_test ,y_test, batch_size=1 )

print("loss : ", loss)
print("acc : ",acc)

output = model.predict(x_test)
print("결과물 : \n", output)

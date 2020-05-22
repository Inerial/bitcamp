import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM


#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[2000,3000,4000],[3000,4000,5000],[4000,5000,6000], [100,200,300]])
y = np.array([4,5,6,7,8,9,10,11,12,13,5000,6000,7000, 400])
x_pred = np.array([55,85,75])

#1개짜리 데이터를 넣을떄 input_dim = 1


from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
x_pred = scaler.transform(x_pred.reshape(1,3))


print(x)
print(x_pred)
x = x.reshape(x.shape[0], x.shape[1], 1)

# 4행 3열짜리 데이터를 한개씩 꺼내쓰겠다는 뜻


## LSTM 함수는 기본적으로 2차원 함수로 출력되

#2. 모델구성
model = Sequential()
model.add(LSTM(800, input_shape=(3,1))) 
# 데이터의 개수인 행은 무시하고 x의 shape
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

model.summary()
## 첫 LSTM layer에 input_shape를 (3,1)을 넣었을때 return_sequences = True를 주지 않으면 리턴시키는 값의 shape가 (batch_size, time_step, feature)의 형태가 아닌
## 기존의 (batch_size, output)의 형태로 나오게 설정되어있다.
## 연속으로 LSTM을 쓰기 위해서는 return_sequences=True를 써주어야 한다. 반대로 Dense를 쓰기 위해서는 다시 false값을 주어야한다(디폴트값)

## one layer에 one bias  다양한 bias가 필요하다면 LSTM을 더 만드는 것이 좋다.

## parameter 계산은 여전히 (input_dim + output + 1) * output * 4 이다.
## 이떄 두번째 이후 input은 이전 LSTM의 output과 같다.

## return_sequences=True값을 준다면 LSTM의 shape형태로 출력되므로 output_shape = (none, 원래 준 time_step, output크기(output노드수 == 다음레이어의 input노드수))


#3. 실행
model.compile(loss = 'mse', optimizer='adam')
model.fit(x,y,epochs=1000)

x_pred = x_pred.reshape(1,3,1)  ## 같은 크기의 행렬로 만들어줌

y_pred = model.predict(x_pred)
print(y_pred)

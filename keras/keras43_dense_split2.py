import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1.data

a = np.array(range(1,101))
size = 5

#LSTM 모델 완성

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:i+size]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)


dataset = split_x(a, size)

x = dataset[:, :size-1]
y = dataset[:, size-1] ## c랑은 다르게 대괄호 안에 ,로 구분한다.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, random_state = 66, train_size = 0.8
)

x_pred = x[-6:]

#2. 모델
model = Sequential()
model.add(Dense(200, input_dim=size-1))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(1))

#3. 훈련
model.compile(optimizer="adam", loss = 'mse',metrics=['mse'])

from keras.callbacks import EarlyStopping
early = EarlyStopping(monitor='val_loss', patience = 15, mode = "auto")

model.fit(x_train, y_train , epochs = 1000, callbacks=[early], validation_split= 0.25)

#4. 평가,예측
loss, mse = model.evaluate(x_test,y_test)
print('loss :', loss)
print('mse :', mse)

y_pred = model.predict(x_pred)
print(y_pred)



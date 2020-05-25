import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1.data

a = np.array(range(1,11))
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

x = dataset.T[:size-1].T.reshape(len(a) - size + 1, size-1, 1)
y = dataset.T[size-1:].T


model = Sequential()
model.add(LSTM(800, input_shape=(size-1,1)))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

model.compile(optimizer="adam", loss = 'mse')

from keras.callbacks import EarlyStopping
early = EarlyStopping(monitor='val_loss', patience = 10, mode = "auto")

model.fit(x, y , epochs = 1000, callbacks=[early])

y_pred = model.predict(x)
print(y_pred)



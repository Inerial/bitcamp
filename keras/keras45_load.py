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

x = dataset[:, :size-1].reshape(len(a) - size + 1, size-1, 1) 
y = dataset[:, size-1] ## c랑은 다르게 대괄호 안에 ,로 구분한다.

from keras.models import load_model
model = load_model("./model/save_keras44.h5")

model.add(Dense(1, name = '1'))

model.summary()

model.compile(optimizer="adam", loss = 'mse',metrics=['mse'])

from keras.callbacks import EarlyStopping
early = EarlyStopping(monitor='val_loss', patience = 5, mode = "auto")

model.fit(x, y , epochs = 1000, callbacks=[early])

loss, mse = model.evaluate(x,y)
print('loss :', loss)
print('mse :', mse)

y_pred = model.predict(x)
print(y_pred)
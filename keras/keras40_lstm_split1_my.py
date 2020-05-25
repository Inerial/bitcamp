import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1.data

a = np.array(range(1,11))
size = 5

#LSTM 모델 완성

def split_x(seq, size):
    import numpy as np
    xxx = []
    yyy = []
    if type(seq) != np.ndarray:
        assert 1 == 2, "입력값이 array가 아님!"
        return
    elif len(seq.shape) == 1:
        for i in range(len(seq) - size + 1):
            subset1 = seq[i:i+size-1]
            subset2 = seq[i+size-1]
            xxx.append(subset1)
            yyy.append(subset2)
        x = np.array(xxx).reshape(len(seq) - size + 1, size-1 , 1)
        y = np.array(yyy)
        return x, y
    elif len(seq.shape) == 2:
        for i in range(len(seq.T) - size + 1):
            subset1 = seq.T[i:i+size - 1]
            subset2 = seq.T[i+size-1]
            xxx.append(subset1)
            yyy.append(subset2)
        x = np.array(xxx)
        y = np.array(yyy)
        return x, y
    else :
        assert 1 == 2 ,"입력값이 3차원 이상!"
        return


x, y = split_x(a, size)


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



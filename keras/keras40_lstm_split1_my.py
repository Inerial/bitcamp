import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1.data

a = np.array(range(1,11))
size = 5

#LSTM 모델 완성

def split_all(seq, size):
    import numpy as np
    aaa = []
    if type(seq) != np.ndarray:
        assert 1 == 2, "입력값이 array가 아님!"
        return
    elif len(seq.shape) == 1:
        for i in range(len(seq) - size + 1):
            subset = seq[i:i+size]
            aaa.append(subset)
        aaa = np.array(aaa).reshape(len(seq) - size + 1, size , 1)
        return aaa
    elif len(seq.shape) == 2:
        for i in range(len(seq.T) - size + 1):
            subset = seq.T[i:i+size]
            aaa.append(subset)
        return np.array(aaa)
    else :
        assert 1 == 2 ,"입력값이 3차원 이상!"
        return


dataset = split_all(a, size)
x = dataset[:, :size-1]  ## c랑은 다르게 대괄호 안에 ,로 구분해서 해준다.
y = dataset[:, size-1]


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



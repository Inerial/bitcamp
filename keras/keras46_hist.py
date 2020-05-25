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


dataset = split_x(a, size).reshape(len(a) - size + 1, size, 1) 

x = dataset[:-6, :size-1]
y = dataset[:-6, size-1] ## c랑은 다르게 대괄호 안에 ,로 구분한다.


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, random_state = 66, train_size = 0.8
)

x_pred = dataset[-6:, :size-1]

# 모델
from keras.models import load_model
model = load_model("./model/save_keras44.h5")

model.add(Dense(100, name = '2'))
model.add(Dense(1, name = '4'))

model.summary()

#훈련
model.compile(optimizer="adam", loss = 'mse',metrics=['acc'])

from keras.callbacks import EarlyStopping

early = EarlyStopping(monitor='val_loss', patience = 20, mode = "auto")

hist = model.fit(x_train, y_train , 
                validation_split=0.25, epochs = 1000, callbacks=[early])

print(hist)
print(hist.history.keys())

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'], label = 'train_loss')
plt.plot(hist.history['val_loss'], label = 'test_loss')
plt.plot(hist.history['acc'], label = 'train_acc')
plt.plot(hist.history['val_acc'], label = 'test_acc')
plt.title('loss&acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
#plt.axis([0,1,0,100])
plt.legend()
plt.show()

#4. 평가,예측
loss, mse = model.evaluate(x_test,y_test)
print('loss :', loss)
print('mse :', mse)

y_pred = model.predict(x_pred)
print(y_pred)


import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

## 1 : 데이터 전처리

x_train = np.load('./data/mnist_train_x.npy')
y_train = np.load('./data/mnist_train_y.npy')
x_test = np.load('./data/mnist_test_x.npy')
y_test = np.load('./data/mnist_test_y.npy')

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

## 2 : 모델링
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(3, (2,2),padding='same', input_shape=(28,28,1)))
model.add(Conv2D(3, (2,2),padding='same'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(3, (2,2),padding='same'))
model.add(Conv2D(3, (2,2),padding='same'))

model.add(Flatten())

model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(10, activation='softmax'))

# model.save('./model/model_test01.h5')
## 이때는 모델만 저장됨


###  3 : 훈련
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

from keras.callbacks import EarlyStopping, ModelCheckpoint
early = EarlyStopping(monitor='loss', patience = 5, mode = 'auto')
modelpath = './model/{epoch:02d}-{val_loss:.2f}.hdf5'
check = ModelCheckpoint(filepath = modelpath, monitor='val_loss', verbose=1,
                        save_best_only=True, ## save best : 최적의 값만 저장
                        save_weights_only=False) ## save weight : 가중치만 저장

hist = model.fit(x_train, y_train, batch_size = 100, epochs=30, validation_split=0.1, callbacks=[early])#, check])

## save한 loss,acc와 load한 loss,acc가 똑같다.
## fit 이후 save시 가중치까지 모델에 저장이 된다.


## 4 : 평가
loss, acc = model.evaluate(x_test,y_test)
print('loss :',loss)
print('acc :',acc)


""" loss : 0.13090835977123425
acc : 0.9800999760627747 """
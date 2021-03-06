import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

## 1 : 데이터 전처리
(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train[0])
print('y_train :',y_train[0])


#plt.imshow(x_train[1], 'gray')
#plt.show()

#from sklearn.preprocessing import OneHotEncoder
#hot = OneHotEncoder()

## OneHotEncoding
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

## 정규화
#from sklearn.preprocessing import MinMaxScaler
x_train = x_train.reshape(60000,28,28,1).astype('float32') / 255
x_test = x_test.reshape(10000,28,28,1).astype('float32') / 255

""" 
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

hist = model.fit(x_train, y_train, batch_size = 100, epochs=30, validation_split=0.1, callbacks=[early, check])
 """
from keras.models import load_model
model = load_model('./model/12-0.07.hdf5')
## save한 loss,acc와 load한 loss,acc가 똑같다.
## fit 이후 save시 가중치까지 모델에 저장이 된다.


## 4 : 평가
loss, acc = model.evaluate(x_test,y_test)
print('loss :',loss)
print('acc :',acc)


""" loss : 0.13090835977123425
acc : 0.9800999760627747 """
from sklearn.datasets import load_boston
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten, Input, LSTM
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os

## 데이터
boston = load_boston()

x = boston.data
y = boston.target

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
x = scale.fit_transform(x)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(
    x,y, random_state=66, train_size = 0.8
)

input1 = Input(shape=(13,))

dense1 = Dense(200, activation='elu')(input1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(180, activation='elu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(160, activation='elu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(150, activation='elu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(130, activation='elu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100, activation='elu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(130, activation='elu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(150, activation='elu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(170, activation='elu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(200, activation='elu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(1, activation='elu')(dense1)

model = Model(inputs=input1, output=dense1)

modelpath = os.path.dirname(os.path.realpath(__file__))


## 모델 세이브 1
model.save(filepath = modelpath + '/boston_model.h5')

model.compile(optimizer='adam', loss = 'mse', metrics=['mse'])

early = EarlyStopping(monitor='val_loss', patience =20)
check = ModelCheckpoint(filepath=modelpath + '/{epoch:02d}-{val_loss:.5f}.hdf5',
                        monitor='val_loss',save_best_only=True, save_weights_only=False)

hist = model.fit(x_train, y_train, batch_size=50, epochs=1000,
                validation_split = 0.3, callbacks = [early, check])

model.save_weights(filepath = modelpath + '/boston_weights.h5')

loss, mse = model.evaluate(x_test, y_test)
print('loss :',loss)
print('mse :',mse)

from sklearn.metrics import r2_score
y_pred = model.predict(x_test)
r2_y = r2_score(y_test,y_pred)
print("결정계수 : ", r2_y)

plt.subplot(2,1,1)
plt.plot(hist.history['loss'], c = 'black',label = 'loss')
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss')
plt.ylabel('loss')
plt.legend()
plt.subplot(2,1,2)
plt.plot(hist.history['mse'], c = 'black',label = 'mse')
plt.plot(hist.history['val_mse'], c = 'blue', label = 'val_mse')
plt.ylabel('mse')
plt.legend()

plt.show() 

""" loss : 14.560861461302814
mse : 14.560861587524414
결정계수 :  0.8271209486595283
 """
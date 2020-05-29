from sklearn.datasets import load_boston
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten, Input, LSTM
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os
tmp = os.getcwd() + '\\keras'
if os.path.isdir(tmp +'\\graph') :
    shutil.rmtree(tmp +'\\graph')
if os.path.isdir(tmp +'\\model') :
    shutil.rmtree(tmp +'\\model')
os.mkdir(tmp +'\\graph')
os.mkdir(tmp +'\\model')

## 데이터
boston = load_boston()

x = boston.data
y = boston.target

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
x = scale.fit_transform(x)

x = x.reshape(x.shape[0], 1, x.shape[1])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(
    x,y, random_state=66, train_size = 0.8
)

input1 = Input(shape=(1, 13))

lstm1 = LSTM(800, activation='elu')(input1)
lstm1 = Dropout(0.2)(lstm1)

dense1 = Dense(200, activation='elu')(lstm1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(200, activation='elu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(200, activation='elu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(200, activation='elu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(200, activation='elu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(200, activation='elu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(1, activation='elu')(dense1)

model = Model(inputs=input1, output=dense1)

model.compile(optimizer='adam', loss = 'mse', metrics=['mse'])

early = EarlyStopping(monitor='val_loss', patience = 10)
tensor = TensorBoard(log_dir = '.\keras\graph', histogram_freq = 0, 
                      write_graph = True, write_images = True)
check = ModelCheckpoint(filepath='.\keras\model\{epoch:02d}-{val_loss:.5f}.hdf5',
                        monitor='val_loss',save_best_only=True)

hist = model.fit(x_train, y_train, batch_size=500, epochs=1000,
                validation_split = 0.3, callbacks = [early, tensor, check])

loss, mse = model.evaluate(x_test, y_test)
print('loss :',loss)
print('mse :',mse)

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

""" loss : 13.677233738057753
mse : 13.677233695983887 """
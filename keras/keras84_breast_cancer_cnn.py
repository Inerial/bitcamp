from sklearn.datasets import load_breast_cancer
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
breast_cancer = load_breast_cancer()

x = breast_cancer.data
y = breast_cancer.target

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
x = scale.fit_transform(x)

x = x.reshape(x.shape[0], 1, 1, x.shape[1])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(
    x,y, random_state=66, train_size = 0.8
)

input1 = Input(shape=(1,1,30))

conv1 = Conv2D(32,(2,2) ,activation='elu', padding = 'same')(input1)
conv1 = Dropout(0.2)(conv1)

conv1 = Flatten()(conv1)

dense1 = Dense(200, activation='elu')(conv1)
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

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['acc'])

early = EarlyStopping(monitor='val_loss', patience =20)
tensor = TensorBoard(log_dir = '.\keras\graph', histogram_freq = 0, 
                      write_graph = True, write_images = True)
check = ModelCheckpoint(filepath='.\keras\model\{epoch:02d}-{val_loss:.5f}.hdf5',
                        monitor='val_loss',save_best_only=True)

hist = model.fit(x_train, y_train, batch_size=500, epochs=1000,
                validation_split = 0.3, callbacks = [early, tensor, check])

loss, acc = model.evaluate(x_test, y_test)
print('loss :',loss)
print('acc :',acc)

plt.subplot(2,1,1)
plt.plot(hist.history['loss'], c = 'black',label = 'loss')
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss')
plt.ylabel('loss')
plt.legend()
plt.subplot(2,1,2)
plt.plot(hist.history['acc'], c = 'black',label = 'acc')
plt.plot(hist.history['val_acc'], c = 'blue', label = 'val_acc')
plt.ylabel('acc')
plt.legend()

plt.show()
""" 
loss : 0.8406731183068794
acc : 0.8157894611358643
"""
from keras.datasets import cifar100
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.layers import Conv2D, Dense, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Input, LSTM
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

(x_train, y_train),(x_test,y_test) = cifar100.load_data()

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

input1 = Input(shape=(32,32,3))
conv1 = Conv2D(32, (3,3), activation='elu', padding='same')(input1)
conv1 = Dropout(0.2)(conv1)
conv1 = Conv2D(32, (3,3), activation='elu', )(conv1)
conv1 = Dropout(0.2)(conv1)
conv1 = Conv2D(32, (3,3), activation='elu', )(conv1)
conv1 = Dropout(0.2)(conv1)

conv1 = MaxPooling2D((2,2))(conv1)

conv1 = Conv2D(64, (3,3), activation='elu', padding='same')(conv1)
conv1 = Dropout(0.2)(conv1)
conv1 = Conv2D(64, (3,3), activation='elu', )(conv1)
conv1 = Dropout(0.2)(conv1)
conv1 = Conv2D(64, (3,3), activation='elu', )(conv1)
conv1 = Dropout(0.2)(conv1)

conv1 = MaxPooling2D((2,2))(conv1)

conv1 = Conv2D(128, (3,3), activation='elu', padding='same')(conv1)
conv1 = Dropout(0.2)(conv1)
conv1 = Conv2D(128, (3,3), activation='elu', padding='same')(conv1)
conv1 = Dropout(0.2)(conv1)
conv1 = Conv2D(128, (3,3), activation='elu', padding='same')(conv1)
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
dense1 = Dense(100, activation='softmax')(dense1)

model = Model(inputs=input1, output=dense1)

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['acc'])

early = EarlyStopping(monitor='val_loss', patience = 10)
tensor = TensorBoard(log_dir = '.\keras\graph', histogram_freq = 0, 
                      write_graph = True, write_images = True)
check = ModelCheckpoint(filepath='.\keras\model\{epoch:02d}-{val_loss:.5f}.hdf5',
                        monitor='val_loss',save_best_only=True)

hist = model.fit(x_train, y_train, batch_size=500, epochs=60,
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
loss : 2.3912230743408203
acc : 0.44620001316070557 """
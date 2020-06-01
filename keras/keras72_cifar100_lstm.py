from keras.datasets import cifar100
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten, Input, LSTM
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
## 리턴될 폴더 지우고 다시 생성해 안 비우기
import shutil
import os
tmp = os.getcwd() + '\\keras'
if os.path.isdir(tmp +'\\graph1') :
    shutil.rmtree(tmp +'\\graph1')

if os.path.isdir(tmp +'\\model1') :
    shutil.rmtree(tmp +'\\model1')


os.mkdir(tmp +'\\graph1')
os.mkdir(tmp +'\\model1')

(x_train, y_train),(x_test,y_test) = cifar100.load_data()

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = x_train.reshape(50000,1,32*32*3).astype('float32') / 255
x_test = x_test.reshape(10000,1,32*32*3).astype('float32') / 255

input1 = Input(shape=(1,32*32*3))

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
dense1 = Dense(100, activation='softmax')(dense1)

model = Model(inputs=input1, output=dense1)

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['acc'])

early = EarlyStopping(monitor='val_loss', patience = 10)
tensor = TensorBoard(log_dir = '.\keras\graph1', histogram_freq = 0, 
                      write_graph = True, write_images = True)
check = ModelCheckpoint(filepath='.\keras\model1\{epoch:02d}-{val_loss:.5f}.hdf5',
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
loss : 3.18756058883667
acc : 0.2354000061750412 """
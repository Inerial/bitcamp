from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.layers import Conv2D, Dense, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Input, LSTM, BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2
import matplotlib.pyplot as plt
import numpy as np

## 리턴될 폴더 지우고 다시 생성해 안 비우기

(x_train, y_train),(x_test,y_test) = cifar10.load_data()

# loss = L1 * reduce_

# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255



## 모델
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='elu', input_shape=(32,32,3), kernel_regularizer=l1(0.001)))
model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='elu', kernel_regularizer=l1(0.001)))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='elu', kernel_regularizer=l1(0.001)))
model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='elu', kernel_regularizer=l1(0.001)))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='elu', kernel_regularizer=l1(0.001)))
model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='elu', kernel_regularizer=l1(0.001)))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

model.add(Flatten())
model.add(Dense(256,activation='elu'))
model.add(Dense(128,activation='elu'))
model.add(Dense(64,activation='elu'))
model.add(Dense(10,activation='softmax'))

model.summary()

## regularizer == 규제 weight값을 건드려서 과적합을 막아봄

## 훈련
model.compile(optimizer = Adam(1e-3), loss = 'sparse_categorical_crossentropy', metrics=['acc'])
## 원핫인코딩을 하지 않고 다중분류 가능

hist = model.fit(x_train, y_train, epochs=20,batch_size=128, verbose=1,
                 validation_split=0.3)

loss_acc = model.evaluate(x_test,y_test)

loss, acc = model.evaluate(x_test, y_test)
print('loss :',loss)
print('acc :',acc)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print(loss_acc)


plt.figure(figsize=(9,5))

plt.subplot(2,1,1)
plt.plot(loss, c = 'black',label = 'loss')
plt.plot(val_loss, c = 'blue', label = 'val_loss')
plt.ylabel('loss')
plt.legend()

plt.subplot(2,1,2)
plt.plot(acc, c = 'black',label = 'acc')
plt.plot(val_acc, c = 'blue', label = 'val_acc')
plt.ylabel('acc')
plt.legend()

plt.show()
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten, Input, LSTM, MaxPool2D
import matplotlib.pyplot as plt
import numpy as np
import os

(x_train, y_train),(x_test,y_test) = cifar10.load_data()

## OneHotEncoding
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

## 정규화
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


input1 = Input(shape = (32,32,3))
Conv1 = Conv2D(32, (3,3),activation='elu', padding='same')(input1)
Conv1 = Dropout(0.2)(Conv1)
Conv1 = Conv2D(32, (3,3),activation='elu', padding='same')(Conv1)
Conv1 = Dropout(0.2)(Conv1)
Conv1 = Conv2D(32, (3,3),activation='relu', padding='same')(Conv1)
Conv1 = Dropout(0.2)(Conv1)

Conv1 = MaxPooling2D((2,2))(Conv1)

Conv1 = Conv2D(64, (3,3),activation='elu', padding='same')(Conv1)
Conv1 = Dropout(0.2)(Conv1)
Conv1 = Conv2D(64, (3,3),activation='elu', padding='same')(Conv1)
Conv1 = Dropout(0.2)(Conv1)
Conv1 = Conv2D(64, (3,3),activation='relu', padding='same')(Conv1)
Conv1 = Dropout(0.2)(Conv1)

Conv1 = MaxPooling2D((2,2))(Conv1)

Conv1 = Conv2D(128, (3,3),activation='elu', padding='same')(Conv1)
Conv1 = Dropout(0.2)(Conv1)
Conv1 = Conv2D(128, (3,3),activation='elu', padding='same')(Conv1)
Conv1 = Dropout(0.2)(Conv1)
Conv1 = Conv2D(128, (3,3),activation='relu', padding='same')(Conv1)
Conv1 = Dropout(0.2)(Conv1)

Conv1 = Flatten()(Conv1)

Dense1 = Dense(100, activation='elu')(Conv1)
Dense1 = Dropout(0.2)(Dense1)
Dense1 = Dense(100, activation='elu')(Dense1)
Dense1 = Dropout(0.2)(Dense1)
Dense1 = Dense(100, activation='elu')(Dense1)
Dense1 = Dropout(0.2)(Dense1)
Dense1 = Dense(100, activation='elu')(Dense1)
Dense1 = Dropout(0.2)(Dense1)
Dense1 = Dense(100, activation='elu')(Dense1)
Dense1 = Dropout(0.2)(Dense1)
Dense1 = Dense(10, activation='softmax')(Conv1)

model = Model(inputs = input1, outputs = Dense1)


modelpath = os.path.dirname(os.path.realpath(__file__))

model.save(filepath = modelpath + '/cifar10_model.h5')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

from keras.callbacks import EarlyStopping, ModelCheckpoint
early = EarlyStopping(monitor='val_loss', patience = 20)
check = ModelCheckpoint(filepath=modelpath+'/{epoch:02d}-{val_loss:.4f}.hdf5', monitor='val_loss',
                        save_best_only=True, save_weights_only= False)

model.fit(x_train, y_train, batch_size = 500, epochs=60, validation_split=0.3, callbacks=[early, check])

model.save_weights(filepath = modelpath + '/cifar10_weights.h5')

loss, acc = model.evaluate(x_test,y_test)
print('loss :',loss)
print('acc :',acc)
""" 
loss : 0.8229247344017029
acc : 0.807200014591217 """
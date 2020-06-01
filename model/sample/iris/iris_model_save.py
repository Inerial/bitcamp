from sklearn.datasets import load_iris
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
iris = load_iris()

x = iris.data
y = iris.target

from keras.utils import np_utils
y = np_utils.to_categorical(y)

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
x = scale.fit_transform(x)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(
    x,y, random_state=66, train_size = 0.8
)

input1 = Input(shape=(4,))

dense1 = Dense(200, activation='elu')(input1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(200, activation='elu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(200, activation='elu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100, activation='elu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100, activation='elu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(50, activation='elu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(50, activation='elu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(3, activation='softmax')(dense1)

model = Model(inputs=input1, output=dense1)

modelpath = os.path.dirname(os.path.realpath(__file__))

model.save(filepath=modelpath + '/iris_model.h5')

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['acc'])

early = EarlyStopping(monitor='val_loss', patience =300)
check = ModelCheckpoint(filepath=modelpath + '\{epoch:02d}-{val_loss:.5f}.hdf5',
                        monitor='val_loss',save_best_only=True, save_weights_only=False)
 
hist = model.fit(x_train, y_train, batch_size=100, epochs=100,
                validation_split = 0.2, callbacks = [early, check])

model.save_weights(filepath=modelpath + '/iris_weights.h5')

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

""" loss : 0.03519194573163986
acc : 1.0
 """
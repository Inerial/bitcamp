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

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
x = scale.fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(8)
x = pca.fit_transform(x)

x = x.reshape(x.shape[0], 2,2,2)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(
    x,y, random_state=66, train_size = 0.8
)

input1 = Input(shape=(2,2,2))

conv1 = Conv2D(32,(2,1) ,activation='elu', padding = 'same')(input1)
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
""" 
loss : 17.777094803604424
mse : 17.7770938873291
결정계수 :  0.7873119144845383
"""
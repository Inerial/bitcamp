from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten, Input, LSTM
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
## 데이터

x = np.load('./data/iris_x.npy')
y = np.load('./data/iris_y.npy')

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

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['acc'])

early = EarlyStopping(monitor='val_loss', patience =300)
tensor = TensorBoard(log_dir = '.\keras\graph', histogram_freq = 0, 
                      write_graph = True, write_images = True)
check = ModelCheckpoint(filepath='.\keras\model\{epoch:02d}-{val_loss:.5f}.hdf5',
                        monitor='val_loss',save_best_only=True)
 
hist = model.fit(x_train, y_train, batch_size=100, epochs=100,
                validation_split = 0.2, callbacks = [early])#, tensor, check])

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
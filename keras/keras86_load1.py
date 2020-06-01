import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train[0])
print('y_train :',y_train[0])


#plt.imshow(x_train[1], 'gray')
#plt.show()

#from sklearn.preprocessing import OneHotEncoder
#hot = OneHotEncoder()

## OneHotEncoding
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

## 정규화
#from sklearn.preprocessing import MinMaxScaler
x_train = x_train.reshape(60000,28,28,1).astype('float32') / 255 ## float 안해줘도 되지 않나?
x_test = x_test.reshape(10000,28,28,1).astype('float32') / 255

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(3, (2,2),padding='same', input_shape=(28,28,1)))
model.add(Conv2D(3, (2,2),padding='same'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(3, (2,2),padding='same'))
model.add(Conv2D(3, (2,2),padding='same'))

model.add(Flatten())

model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(200, activation='elu'))
model.add(Dense(10, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

from keras.callbacks import EarlyStopping
early = EarlyStopping(monitor='loss', patience = 5, mode = 'auto')

hist = model.fit(x_train, y_train, batch_size = 100, epochs=30, validation_split=0.1, callbacks=[early])

model.save('./model/model_test01.h5')

loss, acc = model.evaluate(x_test,y_test)
print('loss :',loss)
print('acc :',acc)


loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('acc :',acc) ## compile시 acc를 입력하면 acc를 입력, accuracy를 입력하면 accuracy를 입력해 주어야 출력이된다.
print('val_acc :',val_acc)
print('loss :',loss)
print('val_loss :',val_loss)
""" 
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
## 매개변수 특징, (행, 열, 채널)
plt. plot(hist.history['loss'],marker='.', c = 'red', label = 'loss')
plt. plot(hist.history['val_loss'],marker='.', c = 'blue', label = 'val_loss')

plt.grid() ## 그래프에 줄긋기

plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['loss', 'val_loss'])
plt.legend(loc = 'upper right') ## 각각에 라벨을 줬기 때문에 위에를 구지 안쓰고 이렇게 적어도 된다.

plt.subplot(2,1,2)

plt. plot(hist.history['acc'])
plt. plot(hist.history['val_acc'])

plt.grid() ## 그래프에 줄긋기

plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()
 """
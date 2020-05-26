import numpy as np
from keras.models import Sequential
from keras.layers import Dense


size = 10
x = np.array(range(1,11))
y = np.array([1,2,3,4,5,1,2,3,4,5])
min_y = np.min(y)

## one_hot_encoding
'''
def one_hot(seq, dim):
    new_y = np.zeros((len(seq),dim))
    for i in range(len(seq)):
        new_y[i, seq[i]] = 1
    return new_y

y = one_hot(y, size)
'''
from keras.utils import np_utils
y = np_utils.to_categorical(y - min_y)

#과제 2 10행 5열로 바꾸기


model = Sequential()
model.add(Dense(100,input_dim = 1, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(len(y[0]), activation='softmax'))
## 마지막 activation을 sigmoid같은 0~1사이가 나오는 것을 주어야 한다.

from keras.callbacks import EarlyStopping
early = EarlyStopping(monitor = 'loss', mode= 'auto', patience = 20)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['acc'])
## categorical_crossentropy == 다중분류로 사용 가능한 loss값
## y값의 one_hot_encoding으로 바꿔주어야 작동
'''
[1,2,2,3,5] =>
[1,0,0,0,0]
[0,1,0,0,0]
[0,1,0,0,0]
[0,0,0,1,0]
[0,0,0,0,1]
'''

model.fit(x,y, epochs = 1000, batch_size= 1)

loss, acc = model.evaluate(x,y, batch_size=1)
print('loss :',loss)
print('acc :',acc)

x_pred = np.array([1,2,3,6,7,8])
y_pred = model.predict(x_pred)

## one_hot_decoding
'''
y_predict1 = np.zeros(len(y_pred))
for i in range(len(y_pred)):
    for j in range(len(y_pred[i])):
        y_predict1[i] += j*y_pred[i][j]
y_predict1 += min_y
'''

y_predict2 = []
for i in range(len(y_pred)):
    y_predict2.append(np.argmax(y_pred[i]))
y_predict2 += min_y

print('y_pred :\n', y_pred)
## print('y_predict1 :\n', y_predict1)
print('y_predict2 :\n', y_predict2)
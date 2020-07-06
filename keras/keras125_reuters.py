from keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train),(x_test,y_test) = reuters.load_data(num_words=2000, test_split=0.2, skip_top=20)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

print(x_train[0])
print(y_train[0])

print(len(x_train[0]))

# y의 카테고리 개수 출력
category = np.max(y_train)+1
print("카테고리 :", category) # 46

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)

y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()
print(bbb)
print(bbb.shape)
## 그룹별로 개수 저장
## 그룹바이 사용법 알아두기 - 주간과제


from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)

#2. 모델
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten,Dropout, Conv1D, AveragePooling1D, BatchNormalization, ELU
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2

model = Sequential()
model.add(Embedding(2001, 128))

model.add(Conv1D(256,5,padding='valid'))
model.add(BatchNormalization())
model.add(ELU())
model.add(AveragePooling1D(5))
model.add(Dropout(0.2))


model.add(LSTM(256))
model.add(BatchNormalization())
model.add(ELU())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(ELU())
model.add(Dense(64))
model.add(BatchNormalization())
model.add(ELU())
model.add(Dense(46, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['acc'])
history = model.fit(x_train, y_train, batch_size=500, epochs=100, validation_split=0.2)

acc = model.evaluate(x_test, y_test)[1]

print('acc :', acc)

y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker='.', c='red', label='TestSet Loss')
plt.plot(y_loss, marker='.', c='blue', label='TrainSet Loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
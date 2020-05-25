import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

## 만약에 keras 안에 그래프 폴더가 있다면 삭제해주는 함수(중복 방지)
import shutil
import os
if os.path.isdir(os.getcwd()+'\\graph') :
    shutil.rmtree(os.getcwd()+'\\graph')


#1.data

a = np.array(range(1,101))
size = 5

#LSTM 모델 완성

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:i+size]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)


dataset = split_x(a, size).reshape(len(a) - size + 1, size, 1) 

x = dataset[:-6, :size-1]
y = dataset[:-6, size-1] ## c랑은 다르게 대괄호 안에 ,로 구분한다.


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, random_state = 66, train_size = 0.8
)

x_pred = dataset[-6:, :size-1]

# 모델
model = Sequential()
model.add(LSTM(800, input_shape=(4,1)))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

model.summary()

#훈련
model.compile(optimizer="adam", loss = 'mse',metrics=['acc'])

from keras.callbacks import EarlyStopping, TensorBoard
tb_hist = TensorBoard(log_dir = '.\keras\graph', histogram_freq = 0, 
                      write_graph = True, write_images = True,)

## cmd에서 해당 폴더 들어간후 tensorboard --logdir=. 입력
## http://127.0.0.1:6006/#scalars 그후 이곳으로 들어가면 그래프가 뜸

early = EarlyStopping(monitor='val_loss', patience = 20, mode = "auto")

hist = model.fit(x_train, y_train , 
                validation_split=0.25, epochs = 1000, callbacks=[early, tb_hist])
'''
print(hist)
print(hist.history.keys())

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'], label = 'train_loss')
plt.plot(hist.history['val_loss'], label = 'test_loss')
plt.plot(hist.history['acc'], label = 'train_acc')
plt.plot(hist.history['val_acc'], label = 'test_acc')
plt.title('loss&acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
#plt.axis([0,1,0,100])
plt.legend()
#plt.show()
'''
#4. 평가,예측
loss, mse = model.evaluate(x_test,y_test)
print('loss :', loss)
print('mse :', mse)

y_pred = model.predict(x_pred)
print(y_pred)

## cmd 바로 켜서 명령어 실행해주고 홈페이지까지 열어주는 함수
## ctrl + c가 먹히지 않아 ctrl + break(맨 오른쪽 구석키)로 꺼주어야 한다. <-문제점
## 그냥 끄면 화면만 꺼지고 뒤에서 계속 돌아가는 것을 확인
import os
asdf = os.system("start cmd /c tensorboard --logdir=. " )
os.system('start chrome http://127.0.0.1:6006/#scalars')



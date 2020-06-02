import pandas as pd
import numpy as np
from keras.layers import Dense, LSTM, Dropout, Input, concatenate,GRU, SimpleRNN
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler
import os, shutil, math
## 타임스텝
time_step = 5


## npy 로드
samsung = np.load('./test/samsung.npy', allow_pickle=True)
hite = np.load('./test/hite.npy', allow_pickle=True)

y = samsung[time_step : ]


## 정규화
# scaler = StandardScaler()
# scaler = RobustScaler()
scaler = MaxAbsScaler()
# scaler = MinMaxScaler()
hite_x = scaler.fit_transform(hite)

# scalerS = StandardScaler()
# scalerS = RobustScaler()
scalerS = MaxAbsScaler()
# scalerS = MinMaxScaler()
samsung_x = scalerS.fit_transform(samsung)

## 데이터 분해
def split_all(seq, size):
    import numpy as np
    aaa = []
    if type(seq) != np.ndarray:
        assert 1 == 2, "입력값이 array가 아님!"
        return
    elif len(seq.shape) == 1:
        for i in range(seq.shape[0] - size + 1):
            subset = seq[i:i+size]
            aaa.append(subset)
        aaa = np.array(aaa).reshape(len(seq) - size + 1, size , 1)
        return aaa
    elif len(seq.shape) == 2:
        for i in range(seq.shape[0] - size + 1):
            subset = seq[i:i+size]
            aaa.append(subset)
        return np.array(aaa)
    else :
        assert 1 == 2 ,"입력값이 3차원 이상!"
        return

samsung_split = split_all(samsung_x, time_step)
hite_split = split_all(hite_x, time_step)

## 사용할 데이터들 자르기
## 오늘 y가 없으므로 제외
samsung_x = samsung_split[:-1, :, :]
hite_x = hite_split[:-1, :, :]


## 내일 예측용 데이터
samsung_x_predict = samsung_split[-1, : , :].reshape(1,time_step, samsung.shape[1])
hite_x_predict = hite_split[-1, :, :].reshape(1,time_step, hite.shape[1])

## train_test_split
y_train, y_test , samsung_x_train, samsung_x_test, hite_x_train, hite_x_test = train_test_split(
    y, samsung_x, hite_x, shuffle= True , random_state=66, train_size=0.9
)



tmp = os.path.dirname(os.path.realpath(__file__))

# best모델 불러오기
model = load_model(tmp+'\\' + 'Test0602_kkt_best.hdf5')
## 평가

from sklearn.metrics import r2_score
y_pred = model.predict([hite_x_test,samsung_x_test])
r2_y = r2_score(y_test,y_pred)
print("결정계수 : ", r2_y)

answer = model.predict([hite_x_predict, samsung_x_predict])
print("내일 삼성 시가 :", answer)
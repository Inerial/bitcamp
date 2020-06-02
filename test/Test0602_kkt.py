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

samsung = pd.read_csv('./test/csv/삼성전자 주가.csv', header=0, index_col=0, sep=',', encoding = 'cp949')
hite = pd.read_csv('./test/csv/하이트 주가.csv', header=0, index_col=0, sep=',', encoding = 'cp949')

## index가 없는 행 제거
samsung = samsung[samsung.index.notna()]
hite = hite[hite.index.notna()]
#print(hite.isna()) ## 하필 6월 2일의 하이트 데이터가 빠져있음 => 마지막에 predict를 두번 해보자

## 문자 데이터 숫자로 변환
for i in range(len(samsung.index)):
    samsung.iloc[i,0] = int(samsung.iloc[i,0].replace(',',''))
for i in range(len(hite.index)):
    for j in range(len(hite.iloc[0])):
        if type(hite.iloc[i][j]) == str :
            hite.iloc[i,j] = int(hite.iloc[i,j].replace(',',''))

## 오름차순 정렬
samsung = samsung.sort_values(by = ['일자'], ascending=[True]).values
hite = hite.sort_values(by = ['일자'], ascending=[True]).values


## 별 영향력이 안보이고 NAN값을 처리하기 어려운 거래량 제거
hite = hite[:,:-1]
## NAN 값을 시가 전날에 비해 상승(하락)한 비율만큼 계산하여 채워줌
for i in range(1, hite.shape[1]):
    hite[-1, i] = hite[-2,i] * hite[-1,0] / hite[-2,0]

## npy 저장
np.save('./test/samsung.npy', arr = samsung)
np.save('./test/hite.npy', arr = hite)

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
    y, samsung_x, hite_x, shuffle= True , random_state=66, train_size=0.8
)

## 모델
input1 = Input(shape = (hite_x_train.shape[1], hite_x_train.shape[2]))
lstm1 = GRU(128 ,activation='elu')(input1)
lstm1 = Dropout(0.2)(lstm1)

lstm1 = Dense(32)(lstm1)
lstm1 = Dropout(0.2)(lstm1)
lstm1 = Dense(32)(lstm1)
lstm1 = Dropout(0.2)(lstm1)
lstm1 = Dense(32)(lstm1)
lstm1 = Dropout(0.2)(lstm1)
lstm1 = Dense(32)(lstm1)
lstm1 = Dropout(0.2)(lstm1)
lstm1 = Dense(32)(lstm1)
lstm1 = Dropout(0.2)(lstm1)
lstm1 = Dense(32)(lstm1)
lstm1 = Dropout(0.2)(lstm1)


input2 = Input(shape = (samsung_x_train.shape[1], samsung_x_train.shape[2]))
lstm2 = GRU(256 ,activation='elu')(input2)
lstm2 = Dropout(0.2)(lstm2)

lstm2 = Dense(32)(lstm2)
lstm2 = Dropout(0.2)(lstm2)
lstm2 = Dense(32)(lstm2)
lstm2 = Dropout(0.2)(lstm2)
lstm2 = Dense(32)(lstm2)
lstm2 = Dropout(0.2)(lstm2)
lstm2 = Dense(32)(lstm2)
lstm2 = Dropout(0.2)(lstm2)
lstm2 = Dense(32)(lstm2)
lstm2 = Dropout(0.2)(lstm2)
lstm2 = Dense(32)(lstm2)
lstm2 = Dropout(0.2)(lstm2)

dense1 = concatenate(inputs = [lstm1, lstm2])

dense1 = Dense(1)(dense1)
dense1 = Dropout(0.2)(dense1)

model = Model(inputs=[input1, input2], outputs= dense1)

model.compile(optimizer='adam', loss = 'mse', metrics = ['mse'])

early = EarlyStopping(monitor='val_loss', patience= 30)
check = ModelCheckpoint(filepath = './test/check/{epoch:02d}-{val_loss:.4f}.hdf5'
                        ,save_best_only=True, save_weights_only=False)

model.fit([hite_x_train, samsung_x_train], y_train,
        batch_size= 500, epochs=500, validation_split= 0.4, callbacks=[early,check])

## best값 폴더 밖으로 빼고 나머지 다 지우는 코드
tmp = os.path.dirname(os.path.realpath(__file__))
bestfile = os.listdir(tmp + '\\check')[-1]
shutil.move(tmp+'\\check' + '\\'+ bestfile, tmp + '\\'+ bestfile)
#os.rename(bestfile, 'Test0602_kkt.hdf5')
if os.path.isdir(tmp+'\\check') :
    shutil.rmtree(tmp +'\\check')
os.mkdir(tmp +'\\check')

# best모델 불러오기
model = load_model(tmp+'\\' + bestfile)#Test0602_kkt.hdf5')
## 평가

from sklearn.metrics import r2_score
y_pred = model.predict([hite_x_test,samsung_x_test])
r2_y = r2_score(y_test,y_pred)
print("결정계수 : ", r2_y)

answer = model.predict([hite_x_predict, samsung_x_predict])
print("내일 삼성 시가 :", answer)
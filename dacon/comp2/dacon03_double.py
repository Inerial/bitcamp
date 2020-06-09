''' ## 데이터가 그냥 LSTM하기엔 애매하다
## Conv1D도 가능

## train test data의 각각 id 를 기준으로 375개씩 뽑은다음
## id가 같은것끼리 묶어, LSTM한 후, 376번째 데이터를 유추하여 해당 데이터로 test (정말 유효할까>) 모델을 두번짜게 된다

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv1D, Flatten, MaxPool1D, LSTM
from keras import backend

def kaeri_metric(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: KAERI metric
    '''
    
    return 0.5 * E1(y_true, y_pred) + 0.5 * E2(y_true, y_pred)


### E1과 E2는 아래에 정의됨 ###

def E1(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: distance error normalized with 2e+04
    '''
    
    _t, _p = np.array(y_true)[:,:2], np.array(y_pred)[:,:2]
    
    return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)


def E2(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: sum of mass and velocity's mean squared percentage error
    '''
    
    _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]
    
    
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06)), axis = 1))


x = np.load('./dacon/comp2/x.npy')
y = np.load('./dacon/comp2/y.npy')
x_pred = np.load('./dacon/comp2/x_pred.npy')
time_step = 10

def split_xy(seq, size):
    import numpy as np
    if type(seq) != np.ndarray:
        print("입력값이 array가 아님!")
        return
    elif len(seq.shape) == 1:
        aaa = []
        bbb = []
        for i in range(len(seq) - size + 1):
            subset_x = seq[i:i+size]
            if i != len(seq)-size:
                subset_y = seq[i+size]
            aaa.append(subset_x)
            bbb.append(subset_y)
        aaa, bbb = np.array(aaa), np.array(bbb)
        return aaa.reshape(aaa.shape[0], aaa.shape[1], 1), bbb.reshape(bbb.shape[0], bbb.shape[1],1)
    elif len(seq.shape) == 2:
        aaa = []
        bbb = []
        for i in range(len(seq) - size + 1):
            subset_x = seq[i:i+size]
            if i != len(seq)-size:
                subset_y = seq[i+size]
            aaa.append(subset_x)
            bbb.append(subset_y)
        return np.array(aaa), np.array(bbb)
    else :
        print("입력값이 3차원 이상!")
        return

x_split = []
y_split = []
x_final = []
for i in range(x.shape[0]):
    tmp_x, tmp_y = split_xy(x[i], time_step)
    x_split.append(tmp_x[:-1,:])
    y_split.append(tmp_y[1:, :])
    x_final.append(tmp_x[-1,:].reshape(1, tmp_x[-1,:].shape[0], tmp_x[-1,:].shape[1]))

xs = np.array(x_split)
ys = np.array(y_split)
xs_pred = np.array(x_final)

print(xs.shape)
print(ys.shape)
print(xs_pred.shape)

 
# 2. 1차 모델
x_final = []
for i in range(len(xs)):
    inputs = Input(shape=(xs[i].shape[1], xs[i].shape[2]))
    lstms = LSTM(200)(inputs)

    denses = Dense(25)(lstms)
    denses = Dense(25)(denses)
    denses = Dense(25)(denses)
    denses = Dense(25)(denses)
    denses = Dense(25)(denses)
    outputs = Dense(y.shape[1])(denses)

    model = Model(inputs = inputs, outputs=outputs)

    model.compile(optimizer = 'adam', loss='mse', metrics=['mse'])

    model.fit(xs[i],ys[i],batch_size= 500, epochs = 100, validation_split=0.2)

    y_pred = model.predict(xs_pred[i])
mspe = kaeri_metric(y_test, y_pred)
print('mspe : ', mspe)
 '''
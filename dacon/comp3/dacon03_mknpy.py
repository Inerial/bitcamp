import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

x = pd.read_csv('./data/dacon/comp3/train_features.csv', sep=',', index_col = 0, header = 0)
y = pd.read_csv('./data/dacon/comp3/train_target.csv', sep=',', index_col = 0, header = 0)
x_pred = pd.read_csv('./data/dacon/comp3/test_features.csv', sep=',', index_col = 0, header = 0)

print(x.shape)
print(y.shape)
print(x_pred.shape)

time_step = int(x.shape[0] / y.shape[0])
tmp = []
for i in range(int(x.shape[0]/time_step)):
    tmp.append(x.iloc[i*time_step : (i+1)*time_step, 1:].values)
x_LSTM = np.array(tmp)

tmp = []
for i in range(int(x_pred.shape[0]/time_step)):
    tmp.append(x_pred.iloc[i*time_step : (i+1)*time_step, 1:].values)
x_pred_LSTM = np.array(tmp)

print("===========================")
print(x_LSTM.shape)
print(y.shape)
print(x_pred_LSTM.shape)
print("===========================")

x_fu = []
x_pred_fu = []
for i in range(len(x_LSTM)):
    X1 = x_LSTM[i,:,0]
    Y1 = np.fft.fft(X1)
    P1 = abs(Y1/(375))
    P1[2:-1] = 2*P1[2:-1]
    rank_X1 = np.argsort(P1[:150])[::-1][ :50]
    x_fu.append(rank_X1)

for i in range(len(x_pred_LSTM)):
    X1 = x_pred_LSTM[i,:,0]
    Y1 = np.fft.fft(X1)
    P1 = abs(Y1/(375))
    P1[2:-1] = 2*P1[2:-1]
    rank_X1 = np.argsort(P1[:150])[::-1][ :50]
    x_pred_fu.append(rank_X1)

x_fu = np.array(x_fu).astype('float32')
x_pred_fu = np.array(x_pred_fu).astype('float32')
print(x_fu.shape)
print(x_pred_fu.shape)

np.save('./dacon/comp3/x_lstm.npy', arr=x_LSTM)
np.save('./dacon/comp3/y.npy', arr=y)
np.save('./dacon/comp3/x_pred_lstm.npy', arr=x_pred_LSTM)
np.save('./dacon/comp3/x_fu.npy', arr=x_fu)
np.save('./dacon/comp3/x_pred_fu.npy', arr=x_pred_fu)

## 위아래 진동수 변수로 줄만하지않나?
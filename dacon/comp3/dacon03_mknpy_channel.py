import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import pywt

x = pd.read_csv('./data/dacon/comp3/train_features.csv', sep=',', index_col = 0, header = 0)
y = pd.read_csv('./data/dacon/comp3/train_target.csv', sep=',', index_col = 0, header = 0)
x_pred = pd.read_csv('./data/dacon/comp3/test_features.csv', sep=',', index_col = 0, header = 0)

print(x.shape)
print(y.shape)
print(x_pred.shape)

scaler = StandardScaler()

time_step = int(x.shape[0] / y.shape[0])
tmp = []
for i in range(int(x.shape[0]/time_step)):
    mid = []
    for j in range(4):
        asdf = x.iloc[i*time_step : (i+1)*time_step, j+1].values
        mid.append(asdf)
        mid.append(scaler.fit_transform(asdf.reshape(375,1))[:,0])
        mid.append(np.fft.fft(asdf, n=375*2).real[:375])
        mid.append(np.fft.fft(asdf, n=375*2).imag[:375])
        mid.append(np.fft.fft(scaler.fit_transform(asdf.reshape(375,1))[:,0], n=375*2).real[:375])
        mid.append(np.fft.fft(scaler.fit_transform(asdf.reshape(375,1))[:,0], n=375*2).imag[:375])
        ca, cb = pywt.dwt(asdf, 'db1')
        cat = pywt.threshold(ca, np.std(ca), mode="hard")
        cbt = pywt.threshold(cb, np.std(cb), mode="hard")
        tx = pywt.idwt(cat, cbt, "db1")[:375]
        mid.append(tx)
        # plt.subplot(6,1,1)
        # plt.plot(range(375), asdf)
        # plt.subplot(6,1,2)
        # plt.plot(range(375), np.fft.fft(asdf, n=375*2).real[:375])
        # plt.subplot(6,1,3)
        # plt.plot(range(375), np.fft.fft(asdf, n=375*2).imag[:375])
        # plt.subplot(6,1,4)
        # plt.plot(range(375), np.fft.fft(scaler.fit_transform(asdf.reshape(375,1))[:,0], n=375*2).imag[:375])
        # plt.subplot(6,1,5)
        # plt.plot(range(375), np.fft.fft(scaler.fit_transform(asdf.reshape(375,1))[:,0], n=375*2).imag[:375])
        # plt.subplot(6,1,6)
        # plt.plot(range(375), tx)
        # plt.show()

    tmp.append(np.array(mid).T)

x_LSTM = np.array(tmp)

tmp = []
for i in range(int(x_pred.shape[0]/time_step)):
    mid = []
    for j in range(4):
        asdf = x_pred.iloc[i*time_step : (i+1)*time_step, j+1].values
        mid.append(asdf)
        mid.append(scaler.fit_transform(asdf.reshape(375,1))[:,0])
        mid.append(np.fft.fft(asdf, n=375*2).real[:375])
        mid.append(np.fft.fft(asdf, n=375*2).imag[:375])
        mid.append(np.fft.fft(scaler.fit_transform(asdf.reshape(375,1))[:,0], n=375*2).real[:375])
        mid.append(np.fft.fft(scaler.fit_transform(asdf.reshape(375,1))[:,0], n=375*2).imag[:375])
        ca, cb = pywt.dwt(asdf, 'db1')
        cat = pywt.threshold(ca, np.std(ca), mode="hard")
        cbt = pywt.threshold(cb, np.std(cb), mode="hard")
        tx = pywt.idwt(cat, cbt, "db1")[:375]
        mid.append(tx)

    tmp.append(np.array(mid).T)

x_pred_LSTM = np.array(tmp)

print("===========================")
print(x_LSTM.shape)
print(y.shape)
print(x_pred_LSTM.shape)
print("===========================")


np.save('./dacon/comp3/x_lstm.npy', arr=x_LSTM)
np.save('./dacon/comp3/y.npy', arr=y)
np.save('./dacon/comp3/x_pred_lstm.npy', arr=x_pred_LSTM)
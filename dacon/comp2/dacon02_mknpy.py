import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

x = pd.read_csv('./data/dacon/comp2/train_features.csv', sep=',', index_col = 0, header = 0)
y = pd.read_csv('./data/dacon/comp2/train_target.csv', sep=',', index_col = 0, header = 0)
x_pred = pd.read_csv('./data/dacon/comp2/test_features.csv', sep=',', index_col = 0, header = 0)

print(x.shape)
print(y.shape)
print(x_pred.shape)

time_step = int(x.shape[0] / y.shape[0])
tmp = []
for i in range(int(x.shape[0]/time_step)):
    tmp.append(x.iloc[i*time_step : (i+1)*time_step, 1:].values)
x = np.array(tmp)

tmp = []
for i in range(int(x_pred.shape[0]/time_step)):
    tmp.append(x_pred.iloc[i*time_step : (i+1)*time_step, 1:].values)
x_pred = np.array(tmp)

print(x_pred.shape)



np.save('./dacon/comp2/x.npy', arr=x)
np.save('./dacon/comp2/y.npy', arr=y)
np.save('./dacon/comp2/x_pred.npy', arr=x_pred)

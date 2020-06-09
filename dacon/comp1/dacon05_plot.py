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
from keras.layers import Input, Dense, Dropout
train = pd.read_csv('./data/dacon/comp1/train.csv', sep=',', index_col = 0, header = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', sep=',', index_col = 0, header = 0)

train_col = train.columns[:-4]
test_col = test.columns
y_train_col = train.columns[-4:]
y_train = train.values[:,-4:]
train = train.values[:,:-4]
test = test.values


# scaler = MinMaxScaler()
# train = scaler.fit_transform(train)
# test = scaler.transform(test)

train = pd.DataFrame(train, columns=train_col)
test = pd.DataFrame(test, columns=test_col)

# train[:,1:] = train[:,1:] * train[:,0:1] * train[:,0:1]
# test[:,1:] = test[:,1:] * test[:,0:1] * test[:,0:1]

# # print(train)

train_src = train.filter(regex='_src$',axis=1).T.interpolate().fillna(method ='ffill').fillna(method ='bfill').T.values # 선형보간법
train_dst = train.filter(regex='_dst$',axis=1).T.interpolate().fillna(method ='ffill').fillna(method ='bfill').T.values # 선형보간법
test_src = test.filter(regex='_src$',axis=1).T.interpolate().fillna(method ='ffill').fillna(method ='bfill').T.values
test_dst = test.filter(regex='_dst$',axis=1).T.interpolate().fillna(method ='ffill').fillna(method ='bfill').T.values

x_train ,y_train = np.concatenate([train.values[:,0:1], train_src, train_dst], axis = 1), y_train
x_pred = np.concatenate([test.values[:,0:1], test_src, test_dst], axis = 1)

print(train.filter(regex='_dst$',axis=1))
for i in range(50):
    plt.subplot(2,1,1)
    pd.DataFrame(x_train, columns=train_col).filter(regex='_src$',axis=1).iloc[i, :].plot()
    plt.subplot(2,1,2)
    pd.DataFrame(x_train, columns=train_col).filter(regex='_dst$',axis=1).iloc[i, :].plot()
    plt.show()


# train[:,1:] = train[:,1:] / train[:,0].reshape(train.shape[0],1) / train[:,0].reshape(train.shape[0],1)
# test[:,1:] = test[:,1:] / test[:,0].reshape(test.shape[0],1) / test[:,0].reshape(test.shape[0],1)



# print(train.isnull().sum())
# print(test.isnull().sum())
# for i in range(10):
#     pd.DataFrame(train).filter(regex='_dst$',axis=1).iloc[:500, i].plot()
#     plt.show()
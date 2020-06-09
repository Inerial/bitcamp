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

train = pd.read_csv('./data/dacon/comp1/train.csv', sep=',', index_col = 0, header = 0).fillna(0).values
test = pd.read_csv('./data/dacon/comp1/test.csv', sep=',', index_col = 0, header = 0).fillna(0).values
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', sep=',', index_col = -4, header = 0).fillna(0).values

train[:,1:] = train[:,1:] * train[:,0].reshape(train.shape[0],1) * train[:,0].reshape(train.shape[0],1)
test[:,1:] = test[:,1:] * test[:,0].reshape(test.shape[0],1) * test[:,0].reshape(test.shape[0],1)
# for i in range(train.shape[0]):
#     lens = train.iloc[i,0] ** 2
#     for j in range(1,train.shape[1]-4):
#         print(i,j)
#         if train.iloc[i,j] != np.nan:
#             if train.iloc[i,j] == 0:
#                 train.iloc[i,j] = np.nan
#             else:
#                 train.iloc[i,j] *= lens
        
# for i in range(test.shape[0]):
#     lens = test.iloc[i,0] ** 2
#     for j in range(1,test.shape[1]-4):
#         print(i,j)
#         if test.iloc[i,j] != np.nan:
#             if test.iloc[i,j] == 0:
#                 test.iloc[i,j] = np.nan
#             else:
#                 test.iloc[i,j] *= lens

train = pd.DataFrame(train).interpolate().fillna(method ='ffill').fillna(method ='bfill').values # 선형보간법
test = pd.DataFrame(test).interpolate().fillna(method ='ffill').fillna(method ='bfill').values

# for i in range(10):
#     pd.DataFrame(train).filter(regex='_dst$',axis=1).iloc[:500, i].plot()
#     plt.show()


train[:,1:] = train[:,1:] / train[:,0].reshape(train.shape[0],1) / train[:,0].reshape(train.shape[0],1)
test[:,1:] = test[:,1:] / test[:,0].reshape(test.shape[0],1) / test[:,0].reshape(test.shape[0],1)

# for i in range(train.shape[0]):
#     lens = train.iloc[i,0] ** 2
#     for j in range(1,train.shape[1]):
#         print(i,j)
#         if train.iloc[i,j] != np.nan:
#             train.iloc[i,j] /= lens
        
# for i in range(test.shape[0]):
#     lens = test.iloc[i,0] ** 2
#     for j in range(1,test.shape[1]):
#         print(i,j)
#         if test.iloc[i,j] != np.nan:
#             test.iloc[i,j] /= lens

# print(train.isnull().sum())
# print(test.isnull().sum())
# for i in range(10):
#     pd.DataFrame(train).filter(regex='_dst$',axis=1).iloc[:500, i].plot()
#     plt.show()

x_train ,y_train = train[:,:-4], train[:,-4:]
x_pred = test

print(x_train.shape)
print(y_train.shape)
print(x_pred.shape)

np.save('./dacon/comp1/x_train.npy', arr=x_train)
np.save('./dacon/comp1/y_train.npy', arr=y_train)
np.save('./dacon/comp1/x_pred.npy', arr=x_pred)

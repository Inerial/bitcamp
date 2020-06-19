import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

## 데이터 불러오기
train = pd.read_csv('./data/dacon/comp1/train.csv', sep=',', index_col = 0, header = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', sep=',', index_col = 0, header = 0)

## 데이터 분해 및 columns이름 저장
train_col = train.columns[:-4]
test_col = test.columns
y_train_col = train.columns[-4:]
y_train = train.values[:,-4:]
train = train.values[:,:-4]
test = test.values


# scaler = MinMaxScaler()
# train = scaler.fit_transform(train)
# test = scaler.transform(test)


## NaN값이 있는 train,test값을 다시 데이터 프레임으로 감싸주기
train = pd.DataFrame(train, columns=train_col)
test = pd.DataFrame(test, columns=test_col)


# train[:,1:] = train[:,1:] * train[:,0:1] * train[:,0:1]
# test[:,1:] = test[:,1:] * test[:,0:1] * test[:,0:1]

# # print(train)

## train에서 
train_src = train.filter(regex='_src$',axis=1).T.interpolate().fillna(method ='ffill').fillna(method ='bfill').T.values # 선형보간법
train_dst = train.filter(regex='_dst$',axis=1).T.interpolate().fillna(method ='ffill').fillna(method ='bfill').T.values # 선형보간법
test_src = test.filter(regex='_src$',axis=1).T.interpolate().fillna(method ='ffill').fillna(method ='bfill').T.values
test_dst = test.filter(regex='_dst$',axis=1).T.interpolate().fillna(method ='ffill').fillna(method ='bfill').T.values

train_src_rank = np.argsort(train_src)[::-1][:, :3]
train_dst_rank = np.argsort(train_dst)[::-1][:, :3]
test_src_rank = np.argsort(test_src)[::-1][:, :3]
test_dst_rank = np.argsort(test_dst)[::-1][:, :3]
print(np.array(train_src - train_dst)[:2,:])

x_train = np.concatenate([train.values[:,0:1], train_src, train_dst, train_src - train_dst, train_src_rank, train_dst_rank], axis = 1)
x_pred = np.concatenate([test.values[:,0:1], test_src, test_dst, test_src - test_dst, train_src_rank, train_dst_rank], axis = 1)
# print(pd.DataFrame(np.log10(train_src.values) - np.log10(train_dst.values)))

# print(x_train.shape)
# print(y_train.shape)
# print(x_pred.shape)

np.save('./dacon/comp1/x_train.npy', arr=x_train)
np.save('./dacon/comp1/y_train.npy', arr=y_train)
np.save('./dacon/comp1/x_pred.npy', arr=x_pred)

## 푸리에 변환 
# 섞여버린 파형을 여러개으 순수한 음파로 분해하는 방법

## 이미 이 데이터는 푸리에 변환이 되어있는 데이터이다?
## 각 파장의 세기 그래프 == 푸리에변환
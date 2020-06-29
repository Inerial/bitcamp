import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
import seaborn as sns

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



train_src = train.filter(regex='_src$',axis=1).T.interpolate(limit_direction='both').T.values # 선형보간법
# train_damp = 1
# train_damp = 625/train.values[:,0:1]/train.values[:,0:1]*(10**(625/train.values[:,0:1]/train.values[:,0:1] - 1))
train_damp = np.exp(np.pi*(10 - train.values[:,0:1])/3.44)
# damp_temp = [3.5,3.4,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44,3.44]
# train_damp = np.exp(np.pi*(10 - train.values[:,0:1])/damp_temp)
train_dst = train.filter(regex='_dst$',axis=1).T.interpolate(limit_direction='both').T.values / train_damp# 선형보간법

test_src = test.filter(regex='_src$',axis=1).T.interpolate(limit_direction='both').T.values
# test_damp = 1
# test_damp = 625/test.values[:,0:1]/test.values[:,0:1]*(10**(625/test.values[:,0:1]/test.values[:,0:1] - 1))
test_damp = np.exp(np.pi*(10 - test.values[:,0:1])/3.44)
test_dst = test.filter(regex='_dst$',axis=1).T.interpolate(limit_direction='both').T.values / test_damp




max_test = 0
max_train = 0
train_fu_real = []
train_fu_imag = []
test_fu_real = []
test_fu_imag = []
train_src_dst_fu = []
test_src_dst_fu = []

train_dst_mean = []
test_dst_mean = []

# rho_10 = np.array([0.0 for i in range(35)])
# nrho_10 = 0
# rho_15 = np.array([0.0 for i in range(35)])
# nrho_15 = 0
# rho_20 = np.array([0.0 for i in range(35)])
# nrho_20 = 0
# rho_25 = np.array([0.0 for i in range(35)])
# nrho_25 = 0

scaler = StandardScaler()

times = 70

for i in range(10000):
    tmp_x = 0
    tmp_y = 0
    for j in range(35):
        if train_src[i, j] == 0 and train_dst[i,j] != 0:
            train_src[i,j] = train_dst[i,j]
        if test_src[i, j] == 0 and test_dst[i,j] != 0:
            test_src[i,j] = test_dst[i,j]
    # if train['rho'][i] == 10:
    #     rho_10 += train_dst[i]
    #     nrho_10 += 1
    # if train['rho'][i] == 15:
    #     rho_15 += train_dst[i]
    #     nrho_15 += 1
    # if train['rho'][i] == 20:
    #     rho_20 += train_dst[i]
    #     nrho_20 += 1
    # if train['rho'][i] == 25:
    #     rho_25 += train_dst[i]
    #     nrho_25 += 1
    train_fu_real.append(np.fft.fft(scaler.fit_transform(train_dst[i:i+1].T).T[0], n=times).real[:35]) 
    train_fu_imag.append(np.fft.fft(scaler.fit_transform(train_dst[i:i+1].T).T[0], n=times).imag[:35])
    test_fu_real.append(np.fft.fft(scaler.fit_transform(test_dst[i:i+1].T).T[0], n=times).real[:35])
    test_fu_imag.append(np.fft.fft(scaler.fit_transform(test_dst[i:i+1].T).T[0], n=times).imag[:35])
    train_src_dst_fu.append(np.fft.ifft(train_src[i]-train_dst[i], n=times).real[:35])
    test_src_dst_fu.append(np.fft.ifft(test_src[i]-test_dst[i], n=times).real[:35])
    train_dst_mean.append([train_dst[i].mean()])
    test_dst_mean.append([test_dst[i].mean()])

# print(rho_10)
# print(rho_15)
# print(rho_20)
# print(rho_25)
# print("==========================")
# print(rho_10/nrho_10)
# print(rho_15/nrho_15)
# print(rho_20/nrho_20)
# print(rho_25/nrho_25)
# print("==========================")
# print(rho_25/nrho_25 / (rho_10/nrho_10))
# print(rho_25/nrho_25 / (rho_15/nrho_15))
# print(rho_25/nrho_25 / (rho_20/nrho_20))


# for i in range(10000):
#     if train['rho'][i] == 10:
#         train_dst[i] *= (rho_25/nrho_25 / (rho_10/nrho_10))
#         test_dst[i] *= (rho_25/nrho_25 / (rho_10/nrho_10))
#     if train['rho'][i] == 15:
#         train_dst[i] *= (rho_25/nrho_25 / (rho_15/nrho_15))
#         test_dst[i] *= (rho_25/nrho_25 / (rho_15/nrho_15))
#     if train['rho'][i] == 20:
#         train_dst[i] *= (rho_25/nrho_25 / (rho_20/nrho_20))
#         test_dst[i] *= (rho_25/nrho_25 / (rho_20/nrho_20))

print(train_dst.shape)

trian_dst_mean = np.array(train_dst_mean)
test_dst_mean = np.array(test_dst_mean)


print(max_train)
print(max_test)
print("RHO")
# print(rho_10/nrho_10)
# print(rho_15/nrho_15)
# print(rho_20/nrho_20)
# print(rho_25/nrho_25)





small = 1e-20



x_train = np.concatenate([train.values[:,0:1],trian_dst_mean, train_dst, train_src-train_dst, train_src/(train_dst+small), train_fu_real, train_fu_imag] , axis = 1)
x_pred = np.concatenate([test.values[:,0:1],test_dst_mean, test_dst, test_src-test_dst, test_src/(test_dst+small),test_fu_real,test_fu_imag], axis = 1)


# x_train = np.concatenate([train.values[:,0:1]**2, train_src, train_dst, train_src - train_dst,train_src/(train_dst+small)], axis = 1)
# x_pred = np.concatenate([test.values[:,0:1]**2, test_src, test_dst, test_src - test_dst,test_src/(test_dst+small)], axis = 1)

# x_train = np.concatenate([train.values[:,0:1]**2, train_src, train_dst, train_src - train_dst,train_src/(train_dst+small), np.log10((train_src + small)/(train_dst+small))], axis = 1)
# x_pred = np.concatenate([test.values[:,0:1]**2, test_src, train_dst, test_src - test_dst,test_src/(test_dst+small), np.log10((test_src+small)/(test_dst+small))], axis = 1)
# print(pd.DataFrame(x_train).isnull().sum())
# print(pd.DataFrame(np.log10(train_src.values) - np.log10(train_dst.values)))

print(x_train.shape)
print(y_train.shape)
print(x_pred.shape)

np.save('./dacon/comp1/x_train.npy', arr=x_train)
np.save('./dacon/comp1/y_train.npy', arr=y_train)
np.save('./dacon/comp1/x_pred.npy', arr=x_pred)




train_sns = pd.DataFrame(np.concatenate([x_train, y_train], axis = 1))

plt.figure(figsize=(4,12))
sns.heatmap(train_sns.corr().loc[0:x_train.shape[1], x_train.shape[1]:train_sns.shape[1]].abs())
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
train = pd.read_csv('./data/dacon/comp1/train.csv', sep=',', index_col = 0, header = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', sep=',', index_col = 0, header = 0)

train_col = list(train.columns[:-4])
test_col = list(test.columns)
y_train_col = list(train.columns[-4:])
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

x_train = np.concatenate([train.values[:,0:1], train_src, train_dst, train_src - train_dst], axis = 1)
x_pred = np.concatenate([test.values[:,0:1], test_src, test_dst, test_src - test_dst], axis = 1)

train_col = train_col + [str(i*10+650)+'_diff' for i in range(35)]
test_col = test_col + [str(i*10+650)+'_diff' for i in range(35)]
# print(train.filter(regex='_dst$',axis=1))
for i in range(5):
    plt.subplot(2,2,1)
    pd.DataFrame(x_train, columns=train_col).filter(regex='_src$',axis=1).iloc[i, :].plot()
    plt.subplot(2,2,2)
    pd.DataFrame(x_train, columns=train_col).filter(regex='_dst$',axis=1).iloc[i, :].plot()

    X1 = train_src[i]
    X2 = train_dst[i]
    X1 = np.concatenate([X1])
    X2 = np.concatenate([X2])
    Y1 = np.fft.fft(X1)
    Y2 = np.fft.fft(X2)
    P1 = abs(Y1/35)
    P2 = abs(Y2/35)
    P1[2:-1] = 2*P1[2:-1]
    P2[2:-1] = 2*P2[2:-1]
    f = 1000*np.array(range(0,int(35)))/35

    plt.subplot(2,2,3)
    plt.plot(f,P1)
    plt.subplot(2,2,4)
    plt.plot(f,P2)
    plt.show()

# train[:,1:] = train[:,1:] / train[:,0].reshape(train.shape[0],1) / train[:,0].reshape(train.shape[0],1)
# test[:,1:] = test[:,1:] / test[:,0].reshape(test.shape[0],1) / test[:,0].reshape(test.shape[0],1)



# print(train.isnull().sum())
# print(test.isnull().sum())
# for i in range(10):
#     pd.DataFrame(train).filter(regex='_dst$',axis=1).iloc[:500, i].plot()
#     plt.show()



# def plot_feature_importacnes_cancer(model, title):
#     n_features = x_train.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
#     plt.yticks(np.arange(n_features), train_col)
#     plt.title(title)
#     plt.xlabel("feature_importace")
#     plt.ylabel("Features")
#     plt.ylim(-1, n_features)

# model = DecisionTreeRegressor()
# model.fit(x_train,y_train)
# # print(model.feature_importances_)
# plot_feature_importacnes_cancer(model, "DecisionTree")
# plt.show()

# model = RandomForestRegressor()
# model.fit(x_train,y_train)
# # print(model.feature_importances_)
# plot_feature_importacnes_cancer(model, "RandomForest")
# plt.show()

# for i in range(4):
#     model = GradientBoostingRegressor()
#     model.fit(x_train,y_train[:,i])
#     # print(model.feature_importances_)
#     plt.subplot(2,2,i+1)
#     plot_feature_importacnes_cancer(model, "GradientBoost" + str(i))
# plt.show()

# for i in range(4):
#     model = XGBRegressor(validate_parameters= True, n_jobs= -1, n_estimators= 1000, max_depth= 5, eta= 0.1)
#     model.fit(x_train,y_train[:,i])
#     # print(model.feature_importances_)
#     plt.subplot(2,2,i+1)
#     plot_feature_importacnes_cancer(model, "XGBoost"+str(i))
# plt.show()

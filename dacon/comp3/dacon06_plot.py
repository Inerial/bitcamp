import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

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
    maxs = np.array([x_LSTM[i,:,0].max(), x_LSTM[i,:,1].max(), x_LSTM[i,:,2].max(), x_LSTM[i,:,3].max()])
    mins = np.array([x_LSTM[i,:,0].min(), x_LSTM[i,:,1].min(), x_LSTM[i,:,2].min(), x_LSTM[i,:,3].min()])
    means = np.array([x_LSTM[i,:,0].mean(), x_LSTM[i,:,1].mean(), x_LSTM[i,:,2].mean(), x_LSTM[i,:,3].mean()])
    stds = np.array([x_LSTM[i,:,0].std(), x_LSTM[i,:,1].std(), x_LSTM[i,:,2].std(), x_LSTM[i,:,3].std()])
    medians = np.array([np.median(x_LSTM[i,:,0]), np.median(x_LSTM[i,:,1]), np.median(x_LSTM[i,:,2]), np.median(x_LSTM[i,:,3])])
    # skews = np.array([x_LSTM[i,:,0].skew(), x_LSTM[i,:,1].skew(), x_LSTM[i,:,2].skew(), x_LSTM[i,:,3].skew()])

    X1 = x_LSTM[i,:,0]
    X2 = x_LSTM[i,:,1]
    X3 = x_LSTM[i,:,2]
    X4 = x_LSTM[i,:,3]

    Y1 = np.fft.fft(X1)
    Y2 = np.fft.fft(X2)
    Y3 = np.fft.fft(X3)
    Y4 = np.fft.fft(X4)
    
    P1 = abs(Y1/(375))
    P2 = abs(Y2/(375))
    P3 = abs(Y3/(375))
    P4 = abs(Y4/(375))

    P1[2:-1] = 2*P1[2:-1]
    P2[2:-1] = 2*P2[2:-1]
    P3[2:-1] = 2*P3[2:-1]
    P4[2:-1] = 2*P4[2:-1]
    
    rank_X1 = np.argsort(P1[:150])[::-1][ :3]
    rank_X2 = np.argsort(P2[:150])[::-1][ :3]
    rank_X3 = np.argsort(P3[:150])[::-1][ :3]
    rank_X4 = np.argsort(P4[:150])[::-1][ :3]

    t1 = 0
    for j in range(375):
        if x_LSTM[i,j,0] != 0.0:
            break
        t1+=1
    t2 = 0
    for j in range(375):
        if x_LSTM[i,j,1] != 0.0:
            break
        t2+=1
    t3 = 0
    for j in range(375):
        if x_LSTM[i,j,2] != 0.0:
            break
        t3+=1
    t4 = 0
    for j in range(375):
        if x_LSTM[i,j,3] != 0.0:
            break
        t4+=1
    t_all = np.array([t1,t2,t3,t4])

    all_X = np.concatenate([rank_X1, rank_X2, rank_X3, rank_X4 , maxs, mins, means, stds, medians, t_all/t_all.sum()])
    x_fu.append(all_X)

t1_all,t2_all,t3_all,t4_all = [],[],[],[]
for i in range(len(x_pred_LSTM)):
    maxs = np.array([x_pred_LSTM[i,:,0].max(), x_pred_LSTM[i,:,1].max(), x_pred_LSTM[i,:,2].max(), x_pred_LSTM[i,:,3].max()])
    mins = np.array([x_pred_LSTM[i,:,0].min(), x_pred_LSTM[i,:,1].min(), x_pred_LSTM[i,:,2].min(), x_pred_LSTM[i,:,3].min()])
    means = np.array([x_pred_LSTM[i,:,0].mean(), x_pred_LSTM[i,:,1].mean(), x_pred_LSTM[i,:,2].mean(), x_pred_LSTM[i,:,3].mean()])
    stds = np.array([x_pred_LSTM[i,:,0].std(), x_pred_LSTM[i,:,1].std(), x_pred_LSTM[i,:,2].std(), x_pred_LSTM[i,:,3].std()])
    medians = np.array([np.median(x_pred_LSTM[i,:,0]), np.median(x_pred_LSTM[i,:,1]), np.median(x_pred_LSTM[i,:,2]), np.median(x_pred_LSTM[i,:,3])])
    # skews = np.array([x_pred_LSTM[i,:,0].skew(), x_pred_LSTM[i,:,1].skew(), x_pred_LSTM[i,:,2].skew(), x_pred_LSTM[i,:,3].skew()])

    X1 = x_pred_LSTM[i,:,0]
    X2 = x_pred_LSTM[i,:,1]
    X3 = x_pred_LSTM[i,:,2]
    X4 = x_pred_LSTM[i,:,3]

    Y1 = np.fft.fft(X1)
    Y2 = np.fft.fft(X2)
    Y3 = np.fft.fft(X3)
    Y4 = np.fft.fft(X4)
    
    P1 = abs(Y1/(375))
    P2 = abs(Y2/(375))
    P3 = abs(Y3/(375))
    P4 = abs(Y4/(375))

    P1[2:-1] = 2*P1[2:-1]
    P2[2:-1] = 2*P2[2:-1]
    P3[2:-1] = 2*P3[2:-1]
    P4[2:-1] = 2*P4[2:-1]
    
    rank_X1 = np.argsort(P1[:150])[::-1][ :3]
    rank_X2 = np.argsort(P2[:150])[::-1][ :3]
    rank_X3 = np.argsort(P3[:150])[::-1][ :3]
    rank_X4 = np.argsort(P4[:150])[::-1][ :3]

    t1 = 0
    for j in range(375):
        if x_LSTM[i,j,0] != 0.0:
            break
        t1+=1
    t2 = 0
    for j in range(375):
        if x_LSTM[i,j,1] != 0.0:
            break
        t2+=1
    t3 = 0
    for j in range(375):
        if x_LSTM[i,j,2] != 0.0:
            break
        t3+=1
    t4 = 0
    for j in range(375):
        if x_LSTM[i,j,3] != 0.0:
            break
        t4+=1
    t_all = np.array([t1,t2,t3,t4])

    all_X = np.concatenate([rank_X1, rank_X2, rank_X3, rank_X4, maxs, mins, means, stds, medians, t_all/t_all.sum()])

    x_pred_fu.append(all_X)

x_fu = np.array(x_fu).astype('float32')
x_pred_fu = np.array(x_pred_fu).astype('float32')
print(x_fu.shape)
print(x_pred_fu.shape)
plt.subplot(2,2,1)
plt.hist(x_fu[:,-4])
plt.subplot(2,2,2)
plt.hist(x_fu[:,-3])
plt.subplot(2,2,3)
plt.hist(x_fu[:,-2])
plt.subplot(2,2,4)
plt.hist(x_fu[:,-1])
plt.show()
## 위아래 진동수 변수로 줄만하지않나?
## max, min,   mean,   std,    median
## 최대,최소,   평균, 표준편차 , 중심값

# scaler = StandardScaler()
# x_fu = scaler.fit_transform(x_fu)
# pca = PCA(40)
# x_fu = pca.fit_transform(x_fu)

def plot_feature_importacnes_cancer(model, title, lenth):
    plt.barh(np.arange(lenth), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(lenth))
    plt.title(title)
    plt.xlabel("feature_importace")
    plt.ylabel("Features")
    plt.ylim(-1, lenth)



# for i in range(4):
#     model = XGBRegressor(validate_parameters= True, n_jobs= -1, n_estimators= 1000, max_depth= 5, eta= 0.1)
#     model.fit(x_fu,y.values[:,i])
#     # print(model.feature_importances_)
#     plt.subplot(2,2,i+1)
#     plot_feature_importacnes_cancer(model, y.columns[i])


model = XGBRegressor(validate_parameters= True, n_jobs= -1, n_estimators= 1000, max_depth= 5, eta= 0.1)
model.fit(x_fu[:,:],y.values[:,0])
# print(model.feature_importances_)
plt.subplot(2,2,1)
plot_feature_importacnes_cancer(model, y.columns[0],36)

model = XGBRegressor(validate_parameters= True, n_jobs= -1, n_estimators= 1000, max_depth= 5, eta= 0.1)
model.fit(x_fu[:,:],y.values[:,1])
# print(model.feature_importances_)
plt.subplot(2,2,2)
plot_feature_importacnes_cancer(model, y.columns[1],36)

model = XGBRegressor(validate_parameters= True, n_jobs= -1, n_estimators= 1000, max_depth= 5, eta= 0.1)
model.fit(x_fu[:,:],y.values[:,2])
# print(model.feature_importances_)
plt.subplot(2,2,3)
plot_feature_importacnes_cancer(model, y.columns[2],36)

model = XGBRegressor(validate_parameters= True, n_jobs= -1, n_estimators= 1000, max_depth= 5, eta= 0.1)
model.fit(x_fu[:,:],y.values[:,3])
# print(model.feature_importances_)
plt.subplot(2,2,4)
plot_feature_importacnes_cancer(model, y.columns[3],36)

plt.show()



model = RandomForestRegressor(n_jobs= -1, n_estimators= 1000, max_depth= 5)
model.fit(x_fu[:,:],y.values[:,0])
# print(model.feature_importances_)
plt.subplot(2,2,1)
plot_feature_importacnes_cancer(model, y.columns[0],36)

model = RandomForestRegressor(n_jobs= -1, n_estimators= 1000, max_depth= 5)
model.fit(x_fu[:,:],y.values[:,1])
# print(model.feature_importances_)
plt.subplot(2,2,2)
plot_feature_importacnes_cancer(model, y.columns[1],36)

model = RandomForestRegressor(n_jobs= -1, n_estimators= 1000, max_depth= 5)
model.fit(x_fu[:,:],y.values[:,2])
# print(model.feature_importances_)
plt.subplot(2,2,3)
plot_feature_importacnes_cancer(model, y.columns[2],36)

model = RandomForestRegressor(n_jobs= -1, n_estimators= 1000, max_depth= 5)
model.fit(x_fu[:,:],y.values[:,3])
# print(model.feature_importances_)
plt.subplot(2,2,4)
plot_feature_importacnes_cancer(model, y.columns[3],36)

plt.show()

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

print(x_pred.shape)

for i in range(5):
    X1 = x_LSTM[i,:,0]
    X1 = np.concatenate([X1,X1,X1,X1,X1])
    Y1 = np.fft.fft(X1)
    P1 = abs(Y1/(5*375))
    P1[2:-1] = 2*P1[2:-1]
    rank_X1 = np.argsort(P1[:500])[::-1][ :50]
    print(rank_X1)
    f = 1000*np.array(range(0,int(5*375)))/(5*375)
    plt.subplot(2,1,1)
    plt.plot(x_LSTM[i,:,0])
    plt.subplot(2,1,2)
    plt.plot(f[:500],P1[:500])
    plt.show()













# # model = DecisionTreeRegressor()
# model = RandomForestRegressor()

# model.fit(x_train,y_train)

# print(model.feature_importances_)

# import matplotlib.pyplot as plt
# import numpy as np
# def plot_feature_importacnes_cancer(model):
#     n_features = x_train.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
#     plt.yticks(np.arange(n_features), train_col)
#     plt.xlabel("feature_importace")
#     plt.ylabel("Features")
#     plt.ylim(-1, n_features)

# plot_feature_importacnes_cancer(model)
# plt.show()

# ## 위아래 진동수 변수로 줄만하지않나?
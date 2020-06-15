import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from xgboost import XGBRegressor

def kaeri_metric(y_true, y_pred):
    return 0.5 * E1(y_true, y_pred) + 0.5 * E2(y_true, y_pred)

def E1(y_true, y_pred):
    _t, _p = np.array(y_true)[:,:2], np.array(y_pred)[:,:2] 
    return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)

def E2(y_true, y_pred):
    _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06)), axis = 1))


x = np.load('./dacon/comp3/x_fu.npy')
y = np.load('./dacon/comp3/y.npy')
x_pred = np.load('./dacon/comp3/x_pred_fu.npy')

x_train,x_test,y_train,y_test = train_test_split(
    x,y, train_size=0.8, random_state = 66
)

parameters = {
    'n_estimators' : [1,5,10,20,30,50,100,1000,10000],
    'eta' : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    'max_depth' :[1,2,3,5,10],
    'validate_parameters' : [True, False],
    'n_jobs' : [-1]
}
# 2. 모델
y_test_pred = []
y_pred = []
for i in range(4):
    model = RandomizedSearchCV(XGBRegressor(), parameters, cv=5, n_iter=50)
    # model.fit(x_train, y_train[:,i])
    model.fit(x,y[:,i])
    # print("acc : ",model.score(x_test,y_test[:,i]))

    print(model.best_params_)

    # y_test_pred.append(model.predict(x_test))
    y_pred.append(model.predict(x_pred))
    
y_pred = np.array(y_pred).T
# y_test_pred = np.array(y_test_pred).T
print(y_pred.shape)

# mspe = kaeri_metric(y_test, y_test_pred)
# print('mspe : ', mspe)

submissions = pd.DataFrame({
    "id": range(2800,3500),
    "X": y_pred[:,0],
    "Y": y_pred[:,1],
    "M": y_pred[:,2],
    "V": y_pred[:,3]
})

submissions.to_csv('./dacon/comp3/comp3_sub.csv', index = False)

# acc :  0.9157084134372458
# {'validate_parameters': False, 'n_jobs': -1, 'n_estimators': 1000, 'max_depth': 3, 'eta': 0.2}
# C:\Users\bitcamp\anaconda3\lib\site-packages\xgboost\core.py:444: UserWarning: Use subset (sliced data) of np.ndarray is not recommended because it will generate extra copies and increase memory consumption
#   "because it will generate extra copies and increase " +
# acc :  0.961304588111114
# {'validate_parameters': True, 'n_jobs': -1, 'n_estimators': 1000, 'max_depth': 2, 'eta': 0.4}
# C:\Users\bitcamp\anaconda3\lib\site-packages\xgboost\core.py:444: UserWarning: Use subset (sliced data) of np.ndarray is not recommended because it will generate extra copies and increase memory consumption
#   "because it will generate extra copies and increase " +
# acc :  0.973739901644889
# {'validate_parameters': True, 'n_jobs': -1, 'n_estimators': 1000, 'max_depth': 1, 'eta': 0.6}
# C:\Users\bitcamp\anaconda3\lib\site-packages\xgboost\core.py:444: UserWarning: Use subset (sliced data) of np.ndarray is not recommended because it will generate extra copies and increase memory consumption
#   "because it will generate extra copies and increase " +
# acc :  0.9924472524715161
# {'validate_parameters': False, 'n_jobs': -1, 'n_estimators': 1000, 'max_depth': 2, 'eta': 0.2}
# (700, 4)
# mspe :  0.2127824595636973
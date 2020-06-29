import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from lightgbm import LGBMRegressor
import lightgbm as lgbm

def kaeri_loss():
    def kaeri_metric(y_true, y_pred):
        return "kaeri", 0.5 * E1(y_true, y_pred) + 0.5 * E2(y_true, y_pred), False

    def E1(y_true, y_pred):
        _t, _p = np.array(y_true)[:,:2], np.array(y_pred)[:,:2] 
        return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)

    def E2(y_true, y_pred):
        _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]
        return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06)), axis = 1))
    return kaeri_metric

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

# for i in range(2800):
#     if y[i,0] == 100:
#         print(list(x[i,-4:]))
#         print(x[i,-2]/x[i,-1])
#         print(x[i,-4]/x[i,-3])

x_train,x_test,y_train,y_test = train_test_split(
    x,y, train_size=0.8, random_state = 66
)

def train_model(x_data, y_data, k=5):
    models = []
    
    k_fold = KFold(n_splits=k, shuffle=True, random_state=123)
    
    for train_idx, val_idx in k_fold.split(x_data):
        x_train, y_train = x_data[train_idx,:], y_data[train_idx]
        x_val, y_val = x_data[val_idx,:], y_data[val_idx]
    
        d_train = lgbm.Dataset(data = x_train, label = y_train)
        d_val = lgbm.Dataset(data = x_val, label = y_val)
        
        params = {
            'n_estimators': 5000,
            'learning_rate': 0.8,
            'max_depth': 5, 
            'boosting_type': 'dart', 
            'drop_rate' : 0.3,
            'objective': 'regression', 
            'metric' : 'mse',
            'is_training_metric': True, 
            'num_leaves': 200, 
            'colsample_bytree': 0.7, 
            'subsample': 0.7
            }
        wlist = {'train' : d_train, 'eval': d_val}
        model = lgbm.train(params=params, train_set=d_train, valid_sets=d_val, evals_result=wlist)
        models.append(model)
    
    return models

y_test_pred = []
y_pred = []


models = {}
for label in range(4):
    print('train column : ', label)
    models[label] = train_model(x_train, y_train[:,label], k=10)


y_test_pred = []
y_pred=[]
for col in models:
    test_preds = []
    preds = []
    for model in models[col]:
        test_preds.append(model.predict(x_test))
        preds.append(model.predict(x_pred))
    test_pred = np.mean(test_preds, axis=0)
    pred = np.mean(preds, axis=0)

    y_test_pred.append(test_pred)
    y_pred.append(pred)

y_pred = np.array(y_pred).T
y_test_pred = np.array(y_test_pred).T
print(y_pred.shape)

mspe = kaeri_metric(y_test, y_test_pred)
print('mspe : ', mspe)

submissions = pd.DataFrame({
    "id": range(2800,3500),
    "X": y_pred[:,0],
    "Y": y_pred[:,1],
    "M": y_pred[:,2],
    "V": y_pred[:,3]
})

submissions.to_csv('./dacon/comp3/comp3_sub.csv', index = False)

# mspe :  3.3595243892423294
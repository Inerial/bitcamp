import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from lightgbm import LGBMRegressor
import lightgbm as lgbm
from keras import backend as K 
from keras.layers import Lambda

weight1X = np.array([1,0,0,0])
weight1Y = np.array([0,1,0,0])
weight2M = np.array([0,0,1,0])
weight2V = np.array([0,0,0,1])

def kaeri_metric(y_true, y_pred):
    return 0.5 * E1(y_true, y_pred) + 0.5 * E2(y_true, y_pred)

def E1(y_true, y_pred):
    _t, _p = np.array(y_true)[:,:2], np.array(y_pred)[:,:2] 
    return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)

def E1X(y_true, y_pred):
    _t, _p = np.array(y_true)[:,0:1], np.array(y_pred)[:,0:1] 
    return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)
def E1Y(y_true, y_pred):
    _t, _p = np.array(y_true)[:,1:2], np.array(y_pred)[:,1:2] 
    return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)

def E2(y_true, y_pred):
    _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06)), axis = 1))
def E2M(y_true, y_pred):
    _t, _p = np.array(y_true)[:,2:3], np.array(y_pred)[:,2:3]
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06)), axis = 1))
def E2V(y_true, y_pred):
    _t, _p = np.array(y_true)[:,3:4], np.array(y_pred)[:,3:4]
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06)), axis = 1))

def my_loss(y_true, y_pred):
    divResult = Lambda(lambda x: x[0]/x[1])([(y_pred-y_true),(y_true+0.000001)])
    return K.mean(K.square(divResult))


def my_loss_E1X(y_pred, y_true):
    y_true = y_true.get_label()
    return 'E1X', np.mean(np.square(y_true-y_pred))/2e+04, False

def my_loss_E1Y(y_pred, y_true):
    y_true = y_true.get_label()
    return 'E1Y', np.mean(np.square(y_true-y_pred))/2e+04, False

def my_loss_E2M(y_pred, y_true):
    y_true = y_true.get_label()
    return 'E2M', np.mean(np.square((y_true - y_pred)/(y_true + 1e-06))), False

def my_loss_E2V(y_pred, y_true):
    y_true = y_true.get_label()
    return 'E2V', np.mean(np.square((y_true - y_pred)/(y_true + 1e-06))), False

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

def train_model(x_data, y_data, k=5, metric='mae'):
    models = []
    
    k_fold = KFold(n_splits=k, shuffle=True, random_state=123)
    
    for train_idx, val_idx in k_fold.split(x_data):
        x_train, y_train = x_data[train_idx,:], y_data[train_idx]
        x_val, y_val = x_data[val_idx,:], y_data[val_idx]
    
        d_train = lgbm.Dataset(data = x_train, label = y_train)
        d_val = lgbm.Dataset(data = x_val, label = y_val)
        
        params = {
            'n_estimators': 10000,
            'learning_rate': 0.8,
            'max_depth': 5, 
            'boosting_type': 'dart', 
            'drop_rate' : 0.3,
            'objective': 'regression', 
            # 'metric' : metric,
            'is_training_metric': True, 
            'num_leaves': 200, 
            'colsample_bytree': 0.7, 
            'subsample': 0.7
            }
        wlist = {'train' : d_train, 'eval': d_val}
        model = lgbm.train(params=params, train_set=d_train, valid_sets=d_val, evals_result=wlist, feval= metric)
        models.append(model)
    
    return models

y_test_pred = []
y_pred = []


models = {}
kaeri_metrics = [my_loss_E1X,my_loss_E1Y,my_loss_E2M,my_loss_E2V]
for label in range(4):
    print('train column : ', label)
    models[label] = train_model(x_train, y_train[:,label], k=10, metric=kaeri_metrics[label])


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
e1 = E1(y_test, y_test_pred)
e2 = E2(y_test, y_test_pred)
e2m = E2M(y_test, y_test_pred)
e2v = E2V(y_test, y_test_pred)
print('mspe : ', mspe)
print('E1 : ', e1)
print('E2 : ', e2)
print('E2M : ', e2m)
print('E2V : ', e2v)

submissions = pd.DataFrame({
    "id": range(2800,3500),
    "X": y_pred[:,0],
    "Y": y_pred[:,1],
    "M": y_pred[:,2],
    "V": y_pred[:,3]
})

submissions.to_csv('./dacon/comp3/comp3_sub.csv', index = False)

# mspe :  3.3595243892423294
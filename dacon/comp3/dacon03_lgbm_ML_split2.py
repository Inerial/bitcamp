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
    return np.mean(np.sum(np.square(y_true - y_pred)) / 2e+04)
def E1Y(y_true, y_pred):
    return np.mean(np.sum(np.square(y_true - y_pred)) / 2e+04)

def E2(y_true, y_pred):
    _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06))))
def E2M(y_true, y_pred):
    return np.mean(np.sum(np.square((y_true - y_pred) / (y_true + 1e-06))))
def E2V(y_true, y_pred):
    return np.mean(np.sum(np.square((y_true - y_pred) / (y_true + 1e-06))))

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

def my_split_labels_k(crit):
    s1 = set()
    s1.update(crit)
    s1 = list(s1)
    output = []
    for i in range(len(s1)):
        train_idx = crit != s1[i]
        val_idx = crit == s1[i]
        output.append((train_idx, val_idx))
    return output

# x_train,x_test,y_train,y_test = train_test_split(
#     x,y, train_size=0.8, random_state = 66
# )

ttd = []
for label in range(4):
    kfold = my_split_labels_k(y[:,label])
    if label <= 1:
        x_tr, x_t = x[kfold[0][0], -4:], x[kfold[0][1], -4:]
    else:
        x_tr, x_t = x[kfold[0][0]], x[kfold[0][1]]
    y_tr, y_t = y[kfold[0][0]], y[kfold[0][1]]
    ttd.append({'x_train':x_tr,'x_test':x_t,'y_train':y_tr[:,label],'y_test':y_t[:,label]})

def train_model(x_data, y_data, k=5, metric='mae'):
    models = []
    print(y_data.shape)
    k_fold = my_split_labels_k(y_data)
    
    for train_idx, val_idx in k_fold:
        x_train, y_train = x_data[train_idx,:], y_data[train_idx]
        x_val, y_val = x_data[val_idx,:], y_data[val_idx]
    
        d_train = lgbm.Dataset(data = x_train, label = y_train)
        d_val = lgbm.Dataset(data = x_val, label = y_val)
        
        params = {
            'n_estimators': 1000,
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


models = []
kaeri_metrics = [my_loss_E1X,my_loss_E1Y,my_loss_E2M,my_loss_E2V]
for label in range(4):
    print('train column : ', label)
    models.append(train_model(ttd[label]['x_train'], ttd[label]['y_train'], k=10, metric=kaeri_metrics[label]))


y_pred=[]
EE = [E1X,E1Y,E2M,E2V]

for label, model_list in enumerate(models):

    label_predict = np.zeros(shape=(ttd[label]['y_test'].shape[0],))
    real_predict = np.zeros(shape=(700,))

    for i, model in enumerate(model_list):
        print(label, i)
        label_predict += model.predict(ttd[label]['x_test'])
        real_predict += model.predict(x_pred)
    label_predict /= len(model_list)
    print(ttd[label]['y_test'].shape, label_predict.shape)
    ee = EE[label](ttd[label]['y_test'], label_predict)
    print('E' + str(i) +' :', ee)
    real_predict /= len(model_list)
    y_pred.append(real_predict)

y_pred = np.array(y_pred).T
print(y_pred.shape)


submissions = pd.DataFrame({
    "id": range(2800,3500),
    "X": y_pred[:,0],
    "Y": y_pred[:,1],
    "M": y_pred[:,2],
    "V": y_pred[:,3]
})

submissions.to_csv('./dacon/comp3/comp3_sub.csv', index = False)

# mspe :  3.3595243892423294
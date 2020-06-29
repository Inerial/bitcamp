import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from lightgbm import LGBMRegressor
import lightgbm as lgbm
from sklearn.metrics import r2_score, mean_absolute_error as mae
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
test = pd.read_csv('./data/dacon/comp1/test.csv', sep=',', header = 0, index_col = 0)


x_train = np.load('./dacon/comp1/x_train.npy')
y_train = np.load('./dacon/comp1/y_train.npy')
x_pred = np.load('./dacon/comp1/x_pred.npy')

x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, train_size = 0.8, random_state = 66
)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

# 2. model

def train_model(x_data, y_data, k=5):
    models = []
    scores = []
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
            'metric': 'mae', 
            'is_training_metric': True, 
            'num_leaves': 200, 
            'colsample_bytree': 0.7, 
            'subsample': 0.7
            }
        wlist = {'train' : d_train, 'eval': d_val}
        model = lgbm.train(params=params, train_set=d_train, valid_sets=d_val, evals_result=wlist)
        models.append(model)
        scores.append(mae(y_val, model.predict(x_val)))
    
    return models[np.argmin(scores)]

y_test_pred = []
y_pred = []


models = []
for label in range(4):
    print('train column : ', label)
    models.append(train_model(x_train, y_train[:,label], k=5))


y_test_pred = []
y_pred=[]
for model in models:
    y_test_pred.append(model.predict(x_test))
    y_pred.append(model.predict(x_pred))


y_pred = np.array(y_pred).T
y_test_pred = np.array(y_test_pred).T

r2 = r2_score(y_test,y_test_pred)
mae = mae(y_test,y_test_pred)
print('r2 :', r2)
print('mae :', mae)

# print(features)
# plt.show()

submissions = pd.DataFrame({
    "id": test.index,
    "hhb": y_pred[:,0],
    "hbo2": y_pred[:,1],
    "ca": y_pred[:,2],
    "na": y_pred[:,3]
})

submissions.to_csv('./dacon/comp1/comp1_sub.csv', index = False)


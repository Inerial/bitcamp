import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.decomposition import PCA
from lightgbm import LGBMRegressor
import lightgbm as lgbm
from sklearn.metrics import r2_score, mean_absolute_error as mae
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
test = pd.read_csv('./data/dacon/comp1/test.csv', sep=',', header = 0, index_col = 0)

def train_model(x_data, y_data, k=5):
        models = []
        
        d_train = lgbm.Dataset(data = x_data, label = y_data)
            
        params = {
            'n_estimators': 100,
            'learning_rate': 0.7,
            'boosting_type': 'dart', 
            'drop_rate' : 0.3,
            'objective': 'regression', 
            'metric': 'mae', 
            'is_training_metric': True, 
            'num_leaves': 200, 
            'colsample_bytree': 0.7, 
            'subsample': 0.7
            }
        model = lgbm.train(params=params, train_set=d_train, verbose_eval=1)
        models.append(model)
        
        return models

for pcas in range(178, 0, -1):
    x_train = np.load('./dacon/comp1/x_train.npy')
    y_train = np.load('./dacon/comp1/y_train.npy')
    x_pred = np.load('./dacon/comp1/x_pred.npy')

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_pred = scaler.transform(x_pred)
    pca = PCA(pcas)
    x_train = pca.fit_transform(x_train)
    x_pred = pca.transform(x_pred)

    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, train_size = 0.8, random_state = 66
    )
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)

    # 2. model

    

    y_test_pred = []
    y_pred = []


    models = {}
    for label in range(4):
        models[label] = train_model(x_train, y_train[:,label])


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

    r2 = r2_score(y_test,y_test_pred)
    mae = mae(y_test,y_test_pred)
    print('r2 :', r2)
    print('mae :', mae)

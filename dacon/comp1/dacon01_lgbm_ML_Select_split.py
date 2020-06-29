import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, mean_absolute_error as MAE
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from lightgbm import LGBMRegressor
import lightgbm as lgbm
from random import choice

test = pd.read_csv('./data/dacon/comp1/test.csv', sep=',', header = 0, index_col = 0)

x_train = np.load('./dacon/comp1/x_train.npy')
y_train = np.load('./dacon/comp1/y_train.npy')
x_pred = np.load('./dacon/comp1/x_pred.npy')

x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, train_size = 0.8, random_state = 66
)
print(x_train.shape)
print(y_train.shape)
# print(x_test.shape)

# 2. model
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

def train_model(x_data, y_data, k=5):
    models = []
    
    k_fold = KFold(n_splits=k, shuffle=True, random_state=66)
    for train_idx, val_idx in k_fold.split(x_data):
        x_train, y_train = x_data[train_idx,:], y_data[train_idx]
        x_val, y_val = x_data[val_idx,:], y_data[val_idx]
    
        d_train = lgbm.Dataset(data = x_train, label = y_train)
        d_val = lgbm.Dataset(data = x_val, label = y_val)
        
        wlist = {'train' : d_train, 'eval': d_val}
        model = lgbm.train(params=params, train_set=d_train, valid_sets=d_val, evals_result=wlist)
        models.append(model)
        
    return models

    
final_y_test_pred = []
final_y_pred = []

# 모델 컬럼별 4번
for i in range(4):
    model = LGBMRegressor(**params)
    model.fit(x_train,y_train[:,i], eval_set=[(x_train, y_train[:,i]), (x_test,y_test[:,i])], verbose = True)

    y_test_pred = model.predict(x_test)
    score = model.score(x_test,y_test[:,i])
    mae = MAE(y_test[:,i], y_test_pred)
    print("r2 : ", score)
    print("mae :", mae)

    thresholds = np.sort(model.feature_importances_)[ [i for i in range(0,len(model.feature_importances_), 30)] ]
    print("model.feature_importances_ : ", model.feature_importances_)
    print(thresholds)
    best_mae = mae
    best_model = model
    best_y_pred = model.predict(x_pred)
    best_y_test_pred = y_test_pred

    for thresh in thresholds:
        if(thresh == 0): continue
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                                # median 이 둘중 하나 쓰는거 이해하면 사용 가능
                                                ## 이거 주어준 값 이하의 중요도를 가진 feature를 전부 자르는 파라미터
        select_x_train = selection.transform(x_train)
        select_x_test = selection.transform(x_test)
        select_x_pred = selection.transform(x_pred)

        print(select_x_train.shape)
        
        select_models = train_model(select_x_train, y_train[:,i], k=10)
            
        test_preds = []
        preds = []
        for models in select_models:
            test_preds.append(models.predict(select_x_test))
            preds.append(models.predict(select_x_pred))
        test_pred = np.mean(test_preds, axis=0)
        pred = np.mean(preds, axis=0)

        r2 = r2_score(y_test[:,i],test_pred)
        mae = MAE(y_test[:,i],test_pred)
        if mae <= best_mae:
            print("예아~")
            best_mae = mae
            best_model = select_models
            best_y_pred = pred
            best_y_test_pred = test_pred 
        print("Thresh=%.3f, n=%d, MAE: %.5f R2: %.2f%%" %(thresh, select_x_train.shape[1], mae, r2*100))
    final_y_pred.append(best_y_pred)
    final_y_test_pred.append(best_y_test_pred)


print('MAE :', MAE(y_test, np.array(final_y_test_pred).T))

final_y_pred = np.array(final_y_pred)

submissions = pd.DataFrame({
    "id": test.index,
    "hhb": final_y_pred[0,:],
    "hbo2": final_y_pred[1,:],
    "ca": final_y_pred[2,:],
    "na": final_y_pred[3,:]
})

submissions.to_csv('./dacon/comp1/comp1_sub.csv', index = False)
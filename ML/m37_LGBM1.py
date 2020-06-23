from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
from lightgbm import LGBMClassifier, LGBMRegressor
import time

# boston = load_boston()
# x = boston.data
# y = boston.target

x, y = load_boston(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8, shuffle = True, random_state=66
)

xgb_parameter = [
    {'n_estimators': [10000],
    'learning_rate': [0.001,0.01,0.0025,0.075],
    'max_depth': [4,5,6]},
    {'n_estimators': [10000],
    'learning_rate': [0.001,0.01,0.0025,0.075],
    'colsample_bytree':[0.6,0.68,0.9,1],
    'max_depth': [4,5,6]},
    {'n_estimators': [10000],
    'learning_rate': [0.001,0.01,0.0025,0.075],
    'colsample_bylevel': [0.6,0.68,0.9,1],
    'max_depth': [4,5,6]}
]
xgb_fit_params = {
    'verbose':False,
    'eval_metric': ["logloss","rmse"],
    'eval_set' :[(x_train, y_train), (x_test,y_test)],
    'early_stopping_rounds': 20
}
lgbm_parameter = [
    {'n_estimators': [10000],
    'learning_rate': [0.001,0.01,0.0025,0.075]},
]
lgbm_fit_params = {
    'verbose':False,
    'eval_metric': ["logloss","rmse"],
    'eval_set' :[(x_train, y_train), (x_test,y_test)],
    'early_stopping_rounds': 20
}





#### XGB 셀렉트
start1 = time.time()
model_XGB = XGBRegressor()
model_XGB.fit(x_train,y_train)
score = model_XGB.score(x_test,y_test)
print("r2 : ", score)

thresholds = np.sort(model_XGB.feature_importances_)

print(thresholds)
print(x_train.shape)
print("========================")

best_x_train = x_train
best_x_train = x_test
best_score = score
best_model = model_XGB

for thresh in thresholds:
    selection = SelectFromModel(model_XGB, threshold=thresh, prefit=True)
                                               # median 이 둘중 하나 쓰는거 이해하면 사용 가능
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    print(select_x_train.shape)

    selection_model = RandomizedSearchCV(XGBRegressor(), xgb_parameter, n_jobs=-1, cv = 5, n_iter=1)
    xgb_fit_params['eval_set'] = [(select_x_train, y_train), (select_x_test,y_test)]
    selection_model.fit(select_x_train, y_train, **xgb_fit_params)

    y_pred = selection_model.predict(select_x_test)
    score = r2_score(y_test,y_pred)
    if best_score <= score:
        best_x_train = select_x_train
        best_x_test = select_x_test
        best_score = score
        best_model = selection_model 

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))

import joblib
joblib.dump(best_model, './model/xgb_Save/sfm1-'+str(best_score)+'.dat')


model2 = joblib.load('./model/xgb_Save/sfm1-'+str(best_score)+'.dat')

y_pred = best_model.predict(best_x_test)
r2 = r2_score(y_test,y_pred)
print('r2 :', r2)

end1 = time.time()


import joblib
joblib.dump(best_model, './model/xgb_Save/sfm1-'+str(best_score)+'.dat')
model2 = joblib.load('./model/xgb_Save/sfm1-'+str(best_score)+'.dat')

#### LGBM 셀렉트

start2 = time.time()
model_LGBM = LGBMRegressor()
model_LGBM.fit(x_train,y_train)
score = model_LGBM.score(x_test,y_test)
print("r2 : ", score)

thresholds = np.sort(model_LGBM.feature_importances_)

print(thresholds)
print(x_train.shape)
print("========================")

best_x_train = x_train
best_x_train = x_test
best_score = score
best_model = model_LGBM

for thresh in thresholds:
    selection = SelectFromModel(model_LGBM, threshold=thresh, prefit=True)
                                               # median 이 둘중 하나 쓰는거 이해하면 사용 가능
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    print(select_x_train.shape)

    selection_model = RandomizedSearchCV(LGBMRegressor(), lgbm_parameter, n_jobs=-1, cv = 5, n_iter=1)
    lgbm_fit_params['eval_set'] = [(select_x_train, y_train), (select_x_test,y_test)]
    selection_model.fit(select_x_train, y_train, **lgbm_fit_params)

    y_pred = selection_model.predict(select_x_test)
    score = r2_score(y_test,y_pred)
    if best_score <= score:
        best_x_train = select_x_train
        best_x_test = select_x_test
        best_score = score
        best_model = selection_model 

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))

import joblib
joblib.dump(best_model, './model/lgbm_Save/sfm1-'+str(best_score)+'.dat')
model2 = joblib.load('./model/lgbm_Save/sfm1-'+str(best_score)+'.dat')



y_pred = best_model.predict(best_x_test)
r2 = r2_score(y_test,y_pred)
print('r2 :', r2)

end2 = time.time()




print('XGB : ', end1 - start1)
print('LGBM : ', end2 - start2)
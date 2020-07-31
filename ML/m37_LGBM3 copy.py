from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
from lightgbm import LGBMClassifier, LGBMRegressor
import time


x, y = load_iris(return_X_y=True)

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
    'eval_metric': ["mlogloss","merror"],
    'eval_set' :[(x_train, y_train), (x_test,y_test)],
    'early_stopping_rounds': 20
}
lgbm_parameter = {'n_estimators': [10000],
    'learning_rate': [0.001,0.01,0.0025,0.075],
    'metric': ["multi_logloss"],
    'objective' : 'multiclass_ova'
    }

lgbm_fit_params = {
    'verbose':False,
    'eval_set' :[(x_train, y_train), (x_test,y_test)],
    'early_stopping_rounds': 20
}



#### LGBM 셀렉트

start2 = time.time()
model_LGBM = RandomizedSearchCV(LGBMClassifier(), param_distributions= lgbm_parameter, n_iter=1)
model_LGBM.fit(x_train,y_train)

print(model_LGBM.predict(x_test))
print(y_train)
print(dir(model_LGBM))
print(model_LGBM.n_splits_)
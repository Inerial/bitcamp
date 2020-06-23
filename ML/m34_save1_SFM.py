from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV

# boston = load_boston()
# x = boston.data
# y = boston.target

x, y = load_boston(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8, shuffle = True, random_state=66
)

parameter = [
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
fit_params = {
    'verbose':False,
    'eval_metric': ["logloss","rmse"],
    'eval_set' :[(x_train, y_train), (x_test,y_test)],
    'early_stopping_rounds': 20
}
model = XGBRegressor()
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print("r2 : ", score)


thresholds = np.sort(model.feature_importances_)

print(thresholds)
print(x_train.shape)
print("========================")

best_x_train = x_train
best_x_train = x_test
best_score = score
best_model = model

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                               # median 이 둘중 하나 쓰는거 이해하면 사용 가능
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    print(select_x_train.shape)

    selection_model = RandomizedSearchCV(XGBRegressor(), parameter, n_jobs=-1, cv = 5, n_iter=1)
    fit_params['eval_set'] = [(select_x_train, y_train), (select_x_test,y_test)]
    selection_model.fit(select_x_train, y_train, **fit_params)

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

y_pred = model2.predict(best_x_test)
r2 = r2_score(y_test,y_pred)
print('r2 :', r2)
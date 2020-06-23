import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error as mae, r2_score

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

parameters =[
    {'n_estimators': [1000],
    'learning_rate': [0.025],
    'colsample_bylevel': [0.75],
    'eval_metric': ['mae'],
    'max_depth': [6]
    }
]
kfold = KFold(n_splits=5, shuffle=True, random_state=66)
search = RandomizedSearchCV(XGBRegressor(), parameters, cv = kfold, n_iter=1, n_jobs=-1)
search = MultiOutputRegressor(search)

search.fit(x_train, y_train)

print(search.estimators_)
y_test_pred = search.predict(x_test)

# print(search.best_params_)
print("R2 :", r2_score(y_test,y_test_pred))
print("mae :", mae(y_test,y_test_pred))

y_pred = search.predict(x_pred)
submissions = pd.DataFrame({
    "id": test.index,
    "hhb": y_pred[:,0],
    "hbo2": y_pred[:,1],
    "ca": y_pred[:,2],
    "na": y_pred[:,3]
})

submissions.to_csv('./dacon/comp1/comp1_sub.csv', index = False)

# MAE : 0.3821994561096453
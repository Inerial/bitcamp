import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from xgboost import XGBRFRegressor


def kaeri_metric(y_true, y_pred):
    return 0.5 * E1(y_true, y_pred) + 0.5 * E2(y_true, y_pred)

def E1(y_true, y_pred):
    _t, _p = np.array(y_true)[:,:2], np.array(y_pred)[:,:2] 
    return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)

def E2(y_true, y_pred):
    _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06)), axis = 1))


x = np.load('./dacon/comp2/x_fu.npy')
y = np.load('./dacon/comp2/y.npy')
x_pred = np.load('./dacon/comp2/x_pred_fu.npy')

x_train,x_test,y_train,y_test = train_test_split(
    x,y, train_size=0.8, random_state = 66
)

parameters = {
    'learning_rate' : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95],
    'max_depth' :[5,10,20,30,50,80,100],
    'n_jobs' : [-1]
}
# 2. 모델
y_pred = []
for i in range(4):
    model = RandomizedSearchCV(XGBRFRegressor(), parameters, cv=5, n_iter=50)
    model.fit(x_train, y_train[:,i])

    print("acc : ",model.score(x_test,y_test[:,i]))

    # y_pred = model.predict(x_test)
    # mspe = kaeri_metric(y_test, y_pred)
    # print('mspe : ', mspe)
    print(model.best_params_)
    y_pred.append(model.predict(x_pred))
    
y_pred = np.array(y_pred).T
print(y_pred.shape)

submissions = pd.DataFrame({
    "id": range(2800,3500),
    "X": y_pred[:,0],
    "Y": y_pred[:,1],
    "M": y_pred[:,2],
    "V": y_pred[:,3]
})

submissions.to_csv('./dacon/comp2/comp2_sub.csv', index = False)

# mspe :  3.3595243892423294
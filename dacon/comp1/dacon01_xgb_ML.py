import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from xgboost import XGBRegressor
test = pd.read_csv('./data/dacon/comp1/test.csv', sep=',', header = 0, index_col = 0)

x_train = np.load('./dacon/comp1/x_train.npy')
y_train = np.load('./dacon/comp1/y_train.npy')
x_pred = np.load('./dacon/comp1/x_pred.npy')

# x_train, x_test, y_train, y_test = train_test_split(
#     x_train, y_train, train_size = 0.8, random_state = 66
# )
print(x_train.shape)
print(y_train.shape)
# print(x_test.shape)

# 2. model

parameters = {
    'n_estimators' : [1,5,10,20,30,50,100,1000],
    'eta' : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    'max_depth' :[1,2,3,5,10,100],
    'validate_parameters' : [True, False],
    'n_jobs' : [-1], 'eval_metric' : ['mae']
}
y_pred = []
search = RandomizedSearchCV(XGBRegressor(), parameters, cv = 5, n_iter=50)

for i in range(4):
    search.fit(x_train, y_train[:,i])

    print(search.best_params_)
    # print("MAE :", search.score(x_test,y_test[:,i]))

    y_pred.append(search.predict(x_pred))

y_pred = np.array(y_pred)
submissions = pd.DataFrame({
    "id": test.index,
    "hhb": y_pred[0,:],
    "hbo2": y_pred[1,:],
    "ca": y_pred[2,:],
    "na": y_pred[3,:]
})

submissions.to_csv('./dacon/comp1/comp1_sub.csv', index = False)
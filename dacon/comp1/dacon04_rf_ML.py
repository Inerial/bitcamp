import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import RandomForestRegressor
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

parameters = {
    'criterion':['mae'], 
    'max_depth':[None,1,2,3,4,5],
    'max_features':[None,4,5],
    'max_leaf_nodes':[None,4,5],
    'min_impurity_decrease':[0.0],
    'verbose':[1],
    'n_jobs':[-1]
}

search = RandomizedSearchCV(RandomForestRegressor(), parameters, cv = 5, n_iter=10)
search.fit(x_train, y_train)

# print(search.best_params_)
print("MAE :", search.score(x_test,y_test))
print(model.best_params_)

y_pred = search.predict(x_pred)

y_pred = np.array(y_pred)
submissions = pd.DataFrame({
    "id": test.index,
    "hhb": y_pred[:,0],
    "hbo2": y_pred[:,1],
    "ca": y_pred[:,2],
    "na": y_pred[:,3]
})

submissions.to_csv('./dacon/comp1/comp1_sub.csv', index = False)
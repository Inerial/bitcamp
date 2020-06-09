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
from keras import backend

test = pd.read_csv('./data/dacon/comp1/test.csv', sep=',', header = 0, index_col = 0)

x_train = np.load('./dacon/comp1/x_train.npy')
y_train = np.load('./dacon/comp1/y_train.npy')
x_pred = np.load('./dacon/comp1/x_pred.npy')

x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, train_size = 0.8, random_state = 66
)

# 2. model

def create_hyperparameters():
    n_estimators=[1]
    criterion=['mae']
    max_depth=[None]
    max_features=['auto']
    return {"models__n_estimators" : n_estimators, "models__criterion": criterion, "models__max_depth": max_depth, "models__max_features" : max_features}
parameters = create_hyperparameters()

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('models', RandomForestRegressor(verbose= 1))
])
search = RandomizedSearchCV(pipe, parameters, cv=5, n_iter=1)

search.fit(x_train, y_train)

print(search.best_params_)
print("MAE :", search.score(x_test,y_test))
y_pred = search.predict(x_pred)
submissions = pd.DataFrame({
    "id": test.index,
    "hhb": y_pred[:,0],
    "hbo2": y_pred[:,1],
    "ca": y_pred[:,2],
    "na": y_pred[:,3]
})

submissions.to_csv('./dacon/comp1/comp1_sub.csv', index = False)
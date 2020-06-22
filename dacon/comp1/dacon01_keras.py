import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from xgboost import XGBRegressor
from keras.models import Model
from keras.layers import Dense, Input, Dropout
test = pd.read_csv('./data/dacon/comp1/test.csv', sep=',', header = 0, index_col = 0)

x_train = np.load('./dacon/comp1/x_train.npy')
y_train = np.load('./dacon/comp1/y_train.npy')
x_pred = np.load('./dacon/comp1/x_pred.npy')

x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, train_size = 0.8, random_state = 66
)

input1 = Input(shape=(x_train.shape[1],))

dense1 = Dense(2560)(input1)
dense1 = Dense(2560)(dense1)
dense1 = Dense(2560)(dense1)
dense1 = Dense(2560)(dense1)
dense1 = Dense(2560)(dense1)
output1 = Dense(4)(dense1)

model = Model(inputs=[input1], outputs=[output1])
model.compile(optimizer='adam', loss='mae', metrics=['mae'])

model.fit(x_train,y_train, epochs=100, batch_size=500, validation_split=0.25)




loss, mae = model.evaluate(x_test,y_test)
print(loss)
print(mae)
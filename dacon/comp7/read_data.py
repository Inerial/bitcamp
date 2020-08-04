import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error as mae
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
test = pd.read_csv('./data/dacon/comp7/train.csv', sep=',', header = 0, index_col = 0)
for i in range(1000):
    print(test.values[i,0], test.values[i,1])
    plt.imshow(test.values[i, 2:].reshape(28,28).astype(int))
    plt.show()
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error as mae
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import matplotlib.pyplot as plt
train = pd.read_csv('./data/dacon/comp7/train.csv', sep=',', header = 0, index_col = 0)

x_train = train.values[:, 2:].reshape(-1, 28,28)/255
x_train[x_train < 0.2] = 2
x_train[x_train < 0.55] = 0
x_train[x_train == 2] = 0.2

for i in range(2048):
    if train.values[i,1] != 'B': continue
    print(train.values[i,0], train.values[i,1])
    plt.imshow((x_train[i]*255).reshape(28,28).astype(int),cmap='gray')
    plt.show()

numbers = train.values[:,0] ## 숫자
letters = train.values[:,1] ## 문자

letters = np.array([ord(i)-ord('A') for i in letters])
letters = to_categorical(letters)
# pd.set_option('display.max_row', 500)
# print(train.groupby([ train['letter']]).count().iloc[:,0])
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization,ELU
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from lightgbm import LGBMClassifier
train = pd.read_csv('./data/dacon/comp7/train.csv', sep=',', header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp7/test.csv', sep=',', header = 0, index_col = 0)
submit = pd.read_csv('./data/dacon/comp7/submission.csv', sep=',', header = 0, index_col = 0)

x_train = train.values[:, 2:].reshape(-1, 28*28)/255
x_train[x_train < 0.2] = 2
x_train[x_train < 0.6] = 0
x_train[x_train == 2] = 0.2

y_train = train.values[:,0].astype(int) ## 숫자 
x_train_let = train.values[:,1] ## 문자

from sklearn.decomposition import PCA
pca = PCA(597)
x_train = pca.fit_transform(x_train)
# print(pca.explained_variance_ratio_.sum())
x_train_let = np.array([ord(i)-ord('A') for i in x_train_let])
x_train_let = to_categorical(x_train_let)

x_train = np.concatenate([x_train, x_train_let], axis=1)
print(x_train.shape)

# for i in range(2048):
#     print(train.values[i,0], train.values[i,1])
#     plt.imshow((x_train[i]*255).reshape(28,28).astype(int))
#     plt.show()
# y_train = to_categorical(y_train)
print(y_train.shape)
# y.append(train.values[i,1]) ## 알파벳
    
x_train, x_test, y_train, y_test = train_test_split(
    x_train,y_train, train_size=0.9, shuffle=True, random_state=66
)

x_real = test.values[:, 1:].reshape(-1, 28*28)/255
x_real[x_real < 0.2] = 2
x_real[x_real < 0.60] = 0
x_real[x_real == 2] = 0.2
x_real = pca.transform(x_real)
x_real_let = test.values[:,0] ## 문자

x_real_let = np.array([ord(i)-ord('A') for i in x_real_let])
x_real_let = to_categorical(x_real_let)

x_real = np.concatenate([x_real, x_real_let], axis=1)
print(x_real.shape)


parameter = {
    'n_estimators': 1000,
    'learning_rate': 0.7,
    'max_depth': 10, 
    'boosting_type': 'dart', 
    'drop_rate' : 0.3,
    'objective': 'multiclass', 
    'metric': ['multi_logloss', 'multi_error'], 
    'is_training_metric': True, 
    'num_leaves': 200, 
    'colsample_bytree': 0.7, 
    'subsample': 0.7
    }
fit_params = {
    'verbose': 1,
    'eval_set' : [(x_train,y_train),(x_test,y_test)],
    # 'early_stopping_rounds' : 5
}

model = LGBMClassifier(**parameter)
model.fit(x_train,y_train,**fit_params)

y_pred = model.predict(x_real)

submit['digit'] = y_pred
print(submit)
submit.to_csv('./dacon/comp7/comp7_sub.csv')
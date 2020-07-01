import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, LSTM, BatchNormalization, Lambda
from keras import backend as K 
import keras
weight1 = np.array([2,2,0,0])
weight2 = np.array([0,0,2,2])
weight1X = np.array([4,0,0,0])
weight1Y = np.array([0,4,0,0])
weight2M = np.array([0,0,4,0])
weight2V = np.array([0,0,0,4])
def kaeri_metric(y_true, y_pred):    
    return 0.5 * E1(y_true, y_pred) + 0.5 * E2(y_true, y_pred)
def E1(y_true, y_pred):
    _t, _p = np.array(y_true)[:,:2], np.array(y_pred)[:,:2]
    return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)
def E2(y_true, y_pred):
    _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06)), axis = 1))


def my_loss_E1(y_true, y_pred):
    return K.mean(K.square(y_true-y_pred)*weight1)/2e+04
def my_loss_E1X(y_true, y_pred):
    return K.mean(K.square(y_true-y_pred)*weight1X)/2e+04
def my_loss_E1Y(y_true, y_pred):
    return K.mean(K.square(y_true-y_pred)*weight1Y)/2e+04

def my_loss_E2(y_true, y_pred):
    divResult = Lambda(lambda x: x[0]/x[1])([(y_pred-y_true),(y_true+0.000001)])
    return K.mean(K.square(divResult)*weight2)
def my_loss_E2M(y_true, y_pred):
    divResult = Lambda(lambda x: x[0]/x[1])([(y_pred-y_true),(y_true+0.000001)])
    return K.mean(K.square(divResult)*weight2M)
def my_loss_E2V(y_true, y_pred):
    divResult = Lambda(lambda x: x[0]/x[1])([(y_pred-y_true),(y_true+0.000001)])
    return K.mean(K.square(divResult)*weight2V)


x = np.load('./dacon/comp3/x_lstm.npy')
y = np.load('./dacon/comp3/y.npy')
x_pred = np.load('./dacon/comp3/x_pred_lstm.npy')

x = x.reshape(2800,375,4,1)
x_pred = x_pred.reshape(700,375,4,1)

x_train,x_test,y_train,y_test = train_test_split(
    x,y, train_size=0.8, random_state = 66
)

# 2. 모델
def set_model(my_loss = 'mse'):  # 0:x,y, 1:m, 2:v
    K.clear_session()
    activation = 'elu'
    padding = 'valid'
    model = Sequential()
    nf = 16
    fs = (3,1)

    model.add(Conv2D(nf,fs, padding=padding, activation=activation,input_shape=(375,4,1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(nf*2,fs, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(nf*4,fs, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(nf*8,fs, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(nf*16,fs, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(nf*32,fs, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Flatten())
    model.add(Dense(128, activation ='elu'))
    model.add(Dense(64, activation ='elu'))
    model.add(Dense(32, activation ='elu'))
    model.add(Dense(16, activation ='elu'))
    model.add(Dense(4))

    optimizer = keras.optimizers.Adam()
       
    model.compile(loss=my_loss, optimizer=optimizer)
       
    model.summary()

    return model
    

kaeri_metrics = [my_loss_E1,my_loss_E2]

submit = pd.read_csv('./data/dacon/comp3/sample_submission.csv')
test = pd.read_csv('./data/dacon/comp3/sample_submission.csv').iloc[:560]
final_y_pred = []

for i in range(2):
    model = set_model(my_loss= kaeri_metrics[i])
    model.fit(x_train,y_train,batch_size=256,epochs=100, shuffle=True, validation_split=0.2)

    score = model.evaluate(x_test,y_test)
    print('loss:', score)
    y_pred = model.predict(x_pred)
    y_test_pred = model.predict(x_test)

    if i == 0: # x,y 학습
        submit.iloc[:,1] = y_pred[:,0]
        test.iloc[:,1] = y_test_pred[:,0]
    # elif i == 1:
        test.iloc[:,2] = y_test_pred[:,1]
        submit.iloc[:,2] = y_pred[:,1]

    elif i == 1: # m 학습
        submit.iloc[:,3] = y_pred[:,2]
        test.iloc[:,3] = y_test_pred[:,2]
    # elif i == 3: # m 학습
        test.iloc[:,4] = y_test_pred[:,3]
        submit.iloc[:,4] = y_pred[:,3]


mspe = kaeri_metric(y_test, test.values[:,1:])
print('MSPE : ', mspe)

# submissions = pd.DataFrame({
#     "id": range(2800,3500),
#     "X": y_pred[:,0],
#     "Y": y_pred[:,1],
#     "M": y_pred[:,2],
#     "V": y_pred[:,3]
# })

submit.to_csv('./dacon/comp3/comp3_sub.csv', index = False)
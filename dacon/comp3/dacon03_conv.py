import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, LSTM, BatchNormalization, Lambda, Activation
from keras import backend as K 
from keras.callbacks import ModelCheckpoint
import keras, os
weight1 = np.array([1,1,0,0])
weight2 = np.array([0,0,1,1])
weight1X = np.array([1,0,0,0])
weight1Y = np.array([0,1,0,0])
weight2M = np.array([0,0,1,0])
weight2V = np.array([0,0,0,1])
def kaeri_metric(y_true, y_pred):    
    return 0.5 * E1(y_true, y_pred) + 0.5 * E2(y_true, y_pred)
def E1(y_true, y_pred):
    _t, _p = np.array(y_true)[:,:2], np.array(y_pred)[:,:2]
    return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)
def E1X(y_true, y_pred):
    _t, _p = np.array(y_true)[:,:2], np.array(y_pred)[:,:2]
    return np.mean(np.sum(np.square(_t - _p)*np.array([1,0]), axis = 1) / 2e+04)
def E1Y(y_true, y_pred):
    _t, _p = np.array(y_true)[:,:2], np.array(y_pred)[:,:2]
    return np.mean(np.sum(np.square(_t - _p)*np.array([0,1]), axis = 1) / 2e+04)

def E2(y_true, y_pred):
    _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06)), axis = 1))
def E2M(y_true, y_pred):
    _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06))*np.array([1,0]), axis = 1))
def E2V(y_true, y_pred):
    _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06))*np.array([0,1]), axis = 1))

def my_loss(y_true, y_pred):
    divResult = Lambda(lambda x: x[0]/x[1])([(y_pred-y_true),(y_true+0.000001)])
    return K.mean(K.square(divResult))
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
x_pred = np.load('./dacon/comp3/x_pred_lstm.npy')
x = x.reshape(2800,375,4,7)
x_pred = x_pred.reshape(700,375,4,7)

y = np.load('./dacon/comp3/y.npy')

folder_path = os.path.dirname(os.path.realpath(__file__))

x_train,x_test,y_train,y_test = train_test_split(
    x,y, train_size=0.8, random_state = 66
)

# 2. 모델
def set_model(my_loss, activation = 'elu', nf = 19, fs = (3,1), ps = (2,1), lr = 0.0001, shape= None):  # 0:x,y, 1:m, 2:v
    K.clear_session()

    padding = 'valid'
    model = Sequential()

    model.add(Conv2D(nf,fs, padding=padding,input_shape=shape))#, kernel_regularizer=l2(0.001)))
    model.add(Activation(activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=ps))

    model.add(Conv2D(nf*2,fs, padding=padding))#, kernel_regularizer=l2(0.001)))
    model.add(Activation(activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=ps))

    model.add(Conv2D(nf*4,fs, padding=padding))#, kernel_regularizer=l2(0.001)))
    model.add(Activation(activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=ps))

    model.add(Conv2D(nf*8,fs, padding=padding))#, kernel_regularizer=l2(0.001)))
    model.add(Activation(activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=ps))

    model.add(Conv2D(nf*16,fs, padding=padding))#, kernel_regularizer=l2(0.001)))
    model.add(Activation(activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=ps))

    model.add(Conv2D(nf*32,fs, padding=padding))#, kernel_regularizer=l2(0.001)))
    model.add(Activation(activation))
    model.add(BatchNormalization())


    model.add(Flatten())
    # model.add(Dense(1024, activation='elu'))
    # model.add(Dense(512, activation='elu'))
    # model.add(Dense(256, activation ='elu'))#, kernel_regularizer=l2(0.001)))
    model.add(Dense(4096, activation ='elu'))#, kernel_regularizer=l2(0.001)))
    model.add(Dense(1024, activation ='elu'))#, kernel_regularizer=l2(0.001)))
    model.add(Dense(256, activation ='elu'))#, kernel_regularizer=l2(0.001)))
    model.add(Dense(64, activation ='elu'))#, kernel_regularizer=l2(0.001)))
    model.add(Dense(16, activation ='elu'))#, kernel_regularizer=l2(0.001)))
    model.add(Dense(4))

    optimizer = keras.optimizers.Adam(lr = lr)
       
    model.compile(loss=my_loss, optimizer=optimizer)
       
    model.summary()

    return model
    

kaeri_metrics = [('my_loss_E1',my_loss_E1),('my_loss_E2M',my_loss_E2M),('my_loss_E2V',my_loss_E2V)]

submit = pd.read_csv('./data/dacon/comp3/sample_submission.csv')
test = pd.read_csv('./data/dacon/comp3/sample_submission.csv').iloc[:560]
final_y_pred = []

for i in range(3):
    check = ModelCheckpoint(filepath='./best_model.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
    model = set_model(my_loss= kaeri_metrics[i][1], shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    model.fit(x_train,y_train,batch_size=64,epochs=1000, shuffle=True, validation_split=0.2, callbacks=[check])

    model = load_model('./best_model.hdf5', custom_objects={kaeri_metrics[i][0]:kaeri_metrics[i][1]})

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
    elif i == 2: # m 학습
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
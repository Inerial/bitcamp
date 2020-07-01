import numpy as np, os, shutil
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import mean_absolute_error as mae
from keras import backend as K
import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, LSTM, BatchNormalization, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint

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

def my_split_labels_k(crit):
    s1 = set()
    s1.update(crit)
    s1 = list(s1)
    output = []
    for i in range(len(s1)):
        train_idx = crit != s1[i]
        val_idx = crit == s1[i]
        output.append((train_idx, val_idx))
    return output


def set_model(my_loss):  # 0:x,y, 1:m, 2:v
    K.clear_session()
    activation = 'elu'
    padding = 'same'
    fs = (3,1)

    input1 = Input(shape=(375,4,1))
    conv2 = Conv2D(64,fs, padding=padding, activation=activation)(input1)
    conv2 = BatchNormalization()(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 1))(conv2)

    conv2 = Conv2D(128,fs, padding=padding, activation=activation)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 1))(conv2)

    conv2 = Conv2D(256,fs, padding=padding, activation=activation)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 1))(conv2)

    conv2 = Conv2D(512,fs, padding=padding, activation=activation)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 1))(conv2)

    conv2 = Conv2D(512,fs, padding=padding, activation=activation)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 1))(conv2)

    conv2 = Flatten()(conv2)
    conv2 = Dense(1024, activation ='elu')(conv2)
    conv2 = Dense(256, activation ='elu')(conv2)
    conv2 = Dense(64, activation ='elu')(conv2)
    conv2 = Dense(4)(conv2)

    model = Model(inputs = input1, outputs = conv2)
    optimizer = keras.optimizers.Adam()
       
    model.compile(loss=my_loss, optimizer=optimizer)
       
    model.summary()

    return model
    

def train_model(x_data, y_data, label, batch_size = 32, epochs = 100, metric=('mae', mae), patience=20, name = None):
    model_path = folder_path + '/model_check_15236261'
    best_models_path = folder_path + '/best_models'
    if os.path.isdir(model_path):
        shutil.rmtree(model_path)
    os.mkdir(model_path)

    k_folds = my_split_labels_k(y_data[:,label])

    for train_idx, val_idx in k_folds:
        x_train, y_train = x_data[train_idx,:], y_data[train_idx]
        x_val, y_val = x_data[val_idx,:], y_data[val_idx]
        model = set_model(metric[1])
        
        early = EarlyStopping(monitor='val_loss', patience=patience, mode='auto')
        check = ModelCheckpoint(filepath=model_path + '/{val_loss:.10f}-{epoch:04d}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)

        model.fit(x_train, y_train, batch_size = batch_size, epochs=epochs, validation_data=(x_val,y_val), verbose=1, callbacks=[early, check])
        
    best_model = os.listdir(model_path)[0]

    new_name = name + '.hdf5'
    os.rename(model_path + '/' + best_model, model_path + '/' + new_name)
    shutil.move(model_path + '/'+ new_name , best_models_path + '/' + new_name)

    shutil.rmtree(model_path)

    return

x = np.load('./dacon/comp3/x_lstm.npy')
x_pred = np.load('./dacon/comp3/x_pred_lstm.npy')
x = x.reshape(2800,375,4,1)
x_pred = x_pred.reshape(700,375,4,1)

y = np.load('./dacon/comp3/y.npy')


folder_path = os.path.dirname(os.path.realpath(__file__))

best_models_path = folder_path + '/best_models'

# if os.path.isdir(best_models_path):
#     shutil.rmtree(best_models_path)
# os.mkdir(best_models_path)

# kaeri_metrics = [('my_loss_E1',my_loss_E1),('my_loss_E2',my_loss_E2)]
# kaeri_metrics = [('my_loss_E1',my_loss_E1),('my_loss_E2M',my_loss_E2M),('my_loss_E2V',my_loss_E2V)]
kaeri_metrics = [('my_loss_E1X',my_loss_E1X),('my_loss_E1Y',my_loss_E1Y),('my_loss_E2M',my_loss_E2M),('my_loss_E2V',my_loss_E2V)]

ttd = []
for label in range(4):
    kfold = my_split_labels_k(y[:,label])
    x_tr, x_t = x[kfold[0][0]], x[kfold[0][1]]
    y_tr, y_t = y[kfold[0][0]], y[kfold[0][1]]
    ttd.append({'x_train':x_tr,'x_test':x_t,'y_train':y_tr,'y_test':y_t})


# for label in range(4):
#     print('train column : ', label)
#     train_model(ttd[label]['x_train'], ttd[label]['y_train'],label = label, metric=kaeri_metrics[label], batch_size=512, epochs = 10, patience=300, name = str(label))



submit = pd.read_csv('./data/dacon/comp3/sample_submission.csv')
test = []

for i in range(4):
    print(i)
    # K.clear_session()
    read_name = str(i) + '.hdf5'
    model = load_model(best_models_path + '/' + read_name, custom_objects={kaeri_metrics[i][0]:kaeri_metrics[i][1]})
    test_preds = model.predict(ttd[i]['x_test'])
    preds = model.predict(x_pred)

    if i == 0: # x,y 학습
        submit.iloc[:,1] = preds[:,0]
    elif i == 1:
        submit.iloc[:,2] = preds[:,1]
    elif i == 2: # m 학습
        submit.iloc[:,3] = preds[:,2]
    elif i == 3: # m 학습
        submit.iloc[:,4] = preds[:,3]

    test.append(test_preds)

EE = [E1X,E1Y,E2M,E2V]
for i in range(4):
    ee = EE[i](ttd[i]['y_test'], test[i])
    print('E' + str(i) +' :', ee)



submit.to_csv('./dacon/comp3/comp3_sub.csv', index = False)

# mspe :  3.3595243892423294
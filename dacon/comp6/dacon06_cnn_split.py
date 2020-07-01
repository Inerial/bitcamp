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
from keras.losses import KLDivergence
from scipy.stats import entropy



def set_model():  # 0:x,y, 1:m, 2:v
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
    model.add(Dense(1))

    model.compile(loss=KLDivergence(), optimizer='adam')
       
    model.summary()

    return model
    
x = np.load('./dacon/comp3/x_lstm.npy')a
x_pred = np.load('./dacon/comp3/x_pred_lstm.npy')a
x = x.reshape(2800,375,4,1)
x_pred = x_pred.reshape(700,375,4,1)

x_fu = np.load('./dacon/comp3/x_fu.npy')
y = np.load('./dacon/comp3/y.npy')a
x_pred_fu = np.load('./dacon/comp3/x_pred_fu.npy')a

folder_path = os.path.dirname(os.path.realpath(__file__))

x_train,x_test,y_train,y_test = train_test_split(
    x,y, train_size=0.8, random_state = 66
)

def train_model(x_data, y_data, k=5, batch_size = 32, epochs = 100, patience=20, name = None):
    models = []
    
    k_fold = KFold(n_splits=k, shuffle=True, random_state=123)
    
    for idx, train_val in enumerate(k_fold.split(x_data)):
        train_idx, val_idx = train_val
        x_train, y_train = x_data[train_idx,:], y_data[train_idx]
        x_val, y_val = x_data[val_idx,:], y_data[val_idx]
        model = set_model()

        model_path = folder_path + '/model_check_15236261'
        best_models_path = folder_path + '/best_models'
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        os.mkdir(model_path)
        
        early = EarlyStopping(monitor='val_loss', patience=patience, mode='auto')
        check = ModelCheckpoint(filepath=model_path + '/{epoch:04d}-{val_loss:.10f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)

        model.fit(x_train, y_train, batch_size = batch_size, epochs=epochs, validation_data=(x_val,y_val), verbose=1, callbacks=[early, check])
        
        best_model = os.listdir(model_path)[-1]

        new_name = name + '-' + str(idx)
        os.rename(model_path + '/' + best_model, model_path + '/' + new_name)
        shutil.move(model_path + '/'+ new_name , best_models_path + '/' + new_name)

        shutil.rmtree(model_path)
        
    return models

y_test_pred = []
y_pred = []

best_models_path = folder_path + '/best_models'
if os.path.isdir(best_models_path):
    shutil.rmtree(best_models_path)
os.mkdir(best_models_path)
for label in range(4):
    print('train column : ', label)
    train_model(x_train, y_train[:,label], k=5, batch_size=512, epochs = 10000, patience=300, name = str(label))


y_test_pred = []
y_pred=[]
for i in range(4):
    test_preds = []
    preds = []
    for j in range(2):
        print(i, j)
        read_name = str(i) + '-' + str(j)
        model = load_model(best_models_path + '/' + read_name, custom_objects={'KLDivergence':KLDivergence()})
        test_preds.append(model.predict(x_test)[:,0])
        preds.append(model.predict(x_pred)[:,0])
    test_preds = np.array(test_preds)
    preds = np.array(preds)
    test_pred = np.mean(test_preds, axis=0)
    pred = np.mean(preds, axis=0)

    y_test_pred.append(test_pred)
    y_pred.append(pred)

y_pred = np.array(y_pred).T
y_test_pred = np.array(y_test_pred).T
print(y_pred.shape)

KL = entropy(y_test, y_test_pred)

print('KLD : ', KL)

submissions = pd.DataFrame({
    "id": range(2800,3500),
    "X": y_pred[:,0],
    "Y": y_pred[:,1],
    "M": y_pred[:,2],
    "V": y_pred[:,3]
})

submissions.to_csv('./dacon/comp3/comp3_sub.csv', index = False)

from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.decomposition import PCA
from keras.utils import np_utils
import numpy as np
import pandas as pd
import os, shutil

## 1. 데이터
path = os.path.dirname(os.path.realpath(__file__))
wine = pd.read_csv(path+'\\csv\\winequality-white.csv', sep=';', header = 0, index_col=None)

x = wine[wine.columns[:-1]]
y = wine['quality']
print(x.shape)
print(y.shape)

x = x.values
y = y.values
min_y = min(y)
y = np_utils.to_categorical(y-min_y)

scaler = StandardScaler()
x = scaler.fit_transform(x)
pca = PCA(10)
x = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, random_state = 66, train_size = 0.8
)

# scalerS = StandardScaler()
# scalerS = MinMaxScaler()
# scalerS = MaxAbsScaler()
# scalerS = RobustScaler()
# x_train = scalerS.fit_transform(x_train)
# x_test = scalerS.transform(x_test)


## 2. 모델
input1 = Input(shape = (x_train.shape[1],))

dense1 = Dense(128, activation = 'relu')(input1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(128, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(128, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(128, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(128, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(128, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(128, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(128, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)

dense1 = Dense(y.shape[1], activation = 'softmax')(dense1)

model = Model(inputs=input1, outputs=dense1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
## 3. 훈련

early = EarlyStopping(monitor='val_loss', patience=30)

os.mkdir(path +'\\check')
check = ModelCheckpoint(path+'\\check\\{epoch:02d}-{val_loss:.4f}.hdf5',
                        save_best_only= True, save_weights_only=False)

model.fit(x_train, y_train, epochs=1000, batch_size = 1000,
          validation_split= 0.4, callbacks=[early, check])                              

## 4.평가 예측

bestfile = os.listdir(path+'\\check')[-1]
shutil.move(path+'\\check\\' + bestfile, path + '\\'+ bestfile)
shutil.rmtree(path +'\\check')

model = load_model(path+'\\'+bestfile)

loss, acc = model.evaluate(x_test,y_test)
print('loss : ',loss)
print('acc : ',acc)

# loss :  1.1021028343512087
# acc :  0.5377551317214966
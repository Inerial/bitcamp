import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input, concatenate, Concatenate
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset  = seq[i:(i+size),]
        aaa.append(subset)
    return np.array(aaa)

size = 6


samsung = np.load('./data/samsung.npy', allow_pickle=True)
hite = np.load('./data/hite.npy', allow_pickle=True)

print(samsung.shape) # (509, 1)
print(hite.shape) # (509, 5)

samsung = split_x(samsung, size)
hite = split_x(hite, size)
print(samsung.shape) # (504,6,1)
print(hite.shape) # (504,6,5)

x_sam = samsung[:, 0:5] # (504,5,1)
y_sam = samsung[:, 5] # (504,1)
x_hit = hite[:, 0:5] # (504,5,5)

print(x_sam.shape)
print(y_sam.shape)
print(x_hit.shape)

''' hite = split_x(hite, size)
print(hite.shape)
 '''

input1 = Input(shape=(5,1))
x1 = LSTM(256)(input1)
x1 = Dropout(0.2)(x1)
x1 = Dense(64)(x1)
x1 = Dropout(0.2)(x1)
x1 = Dense(64)(x1)
x1 = Dropout(0.2)(x1)
x1 = Dense(64)(x1)
x1 = Dropout(0.2)(x1)
x1 = Dense(64)(x1)
x1 = Dropout(0.2)(x1)
x1 = Dense(64)(x1)
x1 = Dropout(0.2)(x1)

input2 = Input(shape=(5,5))
x2 = LSTM(256)(input2)
x2 = Dropout(0.2)(x2)
x2 = Dense(64)(x2)
x2 = Dropout(0.2)(x2)
x2 = Dense(64)(x2)
x2 = Dropout(0.2)(x2)
x2 = Dense(64)(x2)
x2 = Dropout(0.2)(x2)
x2 = Dense(64)(x2)
x2 = Dropout(0.2)(x2)

merge = Concatenate()([x1,x2])

output = Dense(1)(merge)

model = Model(inputs=[input1,input2], outputs=output)

model.summary()

#3. 컴파일, 훈련

model.compile(optimizer='adam', loss='mse',metrics=['mse'])
early = EarlyStopping(monitor='val_loss', patience= 20)
model.fit([x_sam, x_hit], y_sam, epochs=500, callbacks= [early], validation_split=0.4)

## 와꾸만 맞춰서 돌아간다.
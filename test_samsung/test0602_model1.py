import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input, concatenate
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

samsung = samsung.reshape(samsung.shape[0],) # (509,)

samsung = split_x(samsung, size)
print(samsung.shape) # (504,6)

x_hit = hite[5:510,:]
x_sam = samsung[:, 0:5]
y_sam = samsung[:, 5]

''' hite = split_x(hite, size)
print(hite.shape)
 '''

input1 = Input(shape=(5,))
x1 = Dense(10)(input1)
x1 = Dense(10)(x1)

input2 = Input(shape=(5,))
x2 = Dense(5)(input2)
x2 = Dense(5)(x2)

merge = concatenate([x1,x2])

output = Dense(1)(merge)

model = Model(inputs=[input1,input2], outputs=output)

model.summary()

#3. 컴파일, 훈련

model.compile(optimizer='adam', loss='mse',metrics=['mse'])
model.fit([x_sam, x_hit], y_sam, epochs=5)

## 와꾸만 맞춰서 돌아간다.
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

y_sam = samsung[size:] # (504,1)

scaler = StandardScaler()
hite = scaler.fit_transform(hite)

scalerS = StandardScaler()
samsung = scalerS.fit_transform(samsung)


pca = PCA(1)
hite = pca.fit_transform(hite)

print(samsung.shape) # (509, 1)
print(hite.shape) # (509, 1)

x_sam = split_x(samsung[:-1,], size)
x_hit = split_x(hite[:-1,], size)

predict_x_sam = x_sam[-1,: ,:].reshape(1,6,1)
predict_x_hit = x_hit[-1,: ,:].reshape(1,6,1)

print(x_sam.shape) # (503, 6, 1)
print(y_sam.shape) # (503, 1)
print(x_hit.shape) # (503, 6, 1)

train_x_sam,test_x_sam,train_x_hit,test_x_hit,train_y_sam,test_y_sam = train_test_split(
    x_sam,x_hit,y_sam, random_state = 66, train_size = 0.8
)
## 여기서 또 scale 적용 가능
## '할수 있다면' train_x 값만 fit한후 tranform들 해주는것이 좋다.
## split한 데이터는 3차원 텐서라 fit가 안되므로 2차원으로 reshape해줬다가 다시 3차원으로 돌려주는 것이 좋다.

train_x_hit_shape = train_x_hit.shape
test_x_hit_shape = test_x_hit.shape
predict_x_hit_shape = predict_x_hit.shape

scalerH = MinMaxScaler()
train_x_hit = scalerH.fit_transform(train_x_hit.reshape(train_x_hit_shape[0], train_x_hit_shape[1]*train_x_hit_shape[2])).reshape(train_x_hit_shape[0], train_x_hit_shape[1],train_x_hit_shape[2])
test_x_hit = scalerH.transform(test_x_hit.reshape(test_x_hit_shape[0], test_x_hit_shape[1]*test_x_hit_shape[2])).reshape(test_x_hit_shape[0], test_x_hit_shape[1],test_x_hit_shape[2])
predict_x_hit = scalerH.transform(predict_x_hit.reshape(predict_x_hit_shape[0], predict_x_hit_shape[1]*predict_x_hit_shape[2])).reshape(predict_x_hit_shape[0], predict_x_hit_shape[1],predict_x_hit_shape[2])


input1 = Input(shape=(size,1))
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

input2 = Input(shape=(size,1))
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
model.fit([train_x_sam, train_x_hit], train_y_sam, epochs=500, callbacks= [early], validation_split=0.3)

loss, mse = model.evaluate([test_x_sam,test_x_hit], test_y_sam)
print('loss :', loss)
print('mse :', mse)

predict_y_sam = model.predict([predict_x_sam, predict_x_hit])

print(predict_y_sam)
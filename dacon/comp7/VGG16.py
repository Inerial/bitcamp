import numpy as np
import pandas as pd
from skimage.transform import resize
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization,ELU, UpSampling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.applications import VGG16
train = pd.read_csv('./data/dacon/comp7/train.csv', sep=',', header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp7/test.csv', sep=',', header = 0, index_col = 0)
submit = pd.read_csv('./data/dacon/comp7/submission.csv', sep=',', header = 0, index_col = 0)

x_train = train.values[:, 2:].reshape(-1, 28,28,1).astype(int)
new_x_train = []
for img in range(len(x_train)):
    new_x_train.append(resize(x_train[img], (56,56)))
x_train = np.array(new_x_train)

y_train = train.values[:,0] ## 숫자 

# for i in range(2048):
#     print(train.values[i,0], train.values[i,1])
#     plt.imshow((x_train[i]*255).reshape(28,28).astype(int))
#     plt.show()
y_train = np_utils.to_categorical(y_train)
# y.append(train.values[i,1]) ## 알파벳
    
x_real = test.values[:, 1:].reshape(-1, 28,28,1).astype(int)
new_x_real = []
for img in range(len(x_real)):
    new_x_real.append(resize(x_real[img], (56,56)))
x_real = np.array(new_x_real)

print(x_real.shape)
input1 = Input(shape=(56,56,1))
conv1 = BatchNormalization()(input1)
conv1 = VGG16(weights=None, include_top =False, input_shape=(56,56,1))(conv1)
conv1 = Flatten()(conv1)
conv1 = Dense(10, activation='softmax')(conv1)

model = Model(inputs=input1, outputs= conv1)
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

check = ModelCheckpoint('./dacon/comp7/bestcheck.hdf5', monitor='val_loss',save_best_only=True)
model.fit(x_train,y_train,batch_size=256, epochs=300, validation_split=0.2, callbacks=[check])

model = load_model('./dacon/comp7/bestcheck.hdf5')
y_pred = model.predict(x_real)

submit['digit'] = np.argmax(y_pred,axis=1)
print(submit)
submit.to_csv('./dacon/comp7/comp7_sub.csv')
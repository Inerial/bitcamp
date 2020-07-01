import os, glob, numpy as np, cv2
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Conv2DTranspose,MaxPooling2D, Dense, Flatten, Dropout, concatenate, AveragePooling2D, BatchNormalization, Add
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from keras.utils.io_utils import HDF5Matrix
from keras.initializers import RandomNormal

folder_path = os.path.dirname(os.path.realpath(__file__))

# 1. 데이터
x_train = HDF5Matrix(folder_path + '/data/data_set.h5', 'x_train', end= 100000)
y_train = HDF5Matrix(folder_path + '/data/data_set.h5', 'y_train', end= 100000)
x_val = HDF5Matrix(folder_path + '/data/data_set.h5', 'x_val', end = 50000)
y_val = HDF5Matrix(folder_path + '/data/data_set.h5', 'y_val', end = 50000)
# x_train_32 = HDF5Matrix(folder_path + '/data/data_set_32.h5', 'x_train_32')
# y_train_32 = HDF5Matrix(folder_path + '/data/data_set_32.h5', 'y_train_32')
# x_val_32 = HDF5Matrix(folder_path + '/data/data_set_32.h5', 'x_val_32')
# y_val_32 = HDF5Matrix(folder_path + '/data/data_set_32.h5', 'y_val_32')

print(x_train.shape)    # (283336, 112, 112, 3)
print(y_train.shape)    # (283336, 112, 112, 3)
print(x_val.shape)      # (106330, 112, 112, 3)
print(y_val.shape)      # (106330, 112, 112, 3)
# print(x_train_32.shape)    # (283336, 32, 32, 3)
# print(y_train_32.shape)    # (283336, 32, 32, 3)
# print(x_val_32.shape)      # (106330, 32, 32, 3)
# print(y_val_32.shape)      # (106330, 32, 32, 3



# 2. 모델
input_x = Input(shape = (112,112,3))
# conv1 = AveragePooling2D((2,2))(input_x)
conv1 = Conv2D(4, (5,5),activation='relu', padding='same')(input_x)
conv1 = BatchNormalization()(conv1)
conv1 = Conv2D(8, (5,5),activation='relu', padding='same')(input_x)
conv1 = BatchNormalization()(conv1)
# conv1 = Conv2D(16, (5,5),activation='relu', padding='same')(conv1)
# conv1 = BatchNormalization()(conv1)

# merge = concatenate([conv1, input_x])
conv1 = Conv2DTranspose(3, (1,1),activation='relu')(conv1)
conv1 = Add()([conv1, input_x])
model = Model(inputs= input_x, outputs = conv1)


# input_x1 = Input(shape = (112,112,3))
# conv1 = Conv2D(8, (5,5),activation='relu', padding='same')(input_x1)
# conv1 = BatchNormalization()(conv1)
# conv1 = Conv2D(16, (5,5),activation='relu', padding='same')(conv1)
# conv1 = BatchNormalization()(conv1)
# # conv1 = Conv2D(32, (5,5),activation='relu', padding='same')(conv1)
# # conv1 = BatchNormalization()(conv1)

# # merge = concatenate([conv1, input_y])
# # conv1 = Conv2D(3, (1,1),activation='relu', padding='same')(merge)
# conv1 = Conv2DTranspose(3, (81,81))(conv1)
# conv1 = Add()([conv1, input_x2])
# model = Model(inputs= [input_x1, input_x2], outputs = conv1)


model.summary()

''' 3. 훈련 '''
# earlystopping & modelcheckpoint
es = EarlyStopping(monitor='val_loss', patience=8, mode='auto')

modelpath = folder_path + '/model_check/{epoch:02d}-{val_loss:.10f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)

model.compile(loss='mae', optimizer='adam', metrics=['mae'])
hist = model.fit(x_train, y_train, epochs=40, batch_size=500, validation_data = (x_val,y_val), shuffle='batch', callbacks=[cp])

## 저장은 modelcheckpoint
#############################################
################## mkh5.py ##################
#############################################

import os, glob, numpy as np, cv2, pandas as pd
from sklearn.model_selection import train_test_split
import h5py

folder_path = os.path.dirname(os.path.realpath(__file__))
train_filepath = folder_path + '/train'
test_filepath = folder_path + '/test'
val_filepath = folder_path + '/validate'
train_list = pd.read_csv(train_filepath + '/train_labels.csv', sep=',', header = None, index_col = None)
test_list = pd.read_csv(test_filepath + '/test_labels.csv', sep=',', header = None, index_col = None)
val_list = pd.read_csv(val_filepath + '/validate_labels.csv', sep=',', header = None, index_col = None)

print(folder_path)

image_w = 112
image_h = 112

with h5py.File(folder_path + '/data/data_set.h5', 'w') as f:
    f.create_dataset('x_train', (train_list.shape[0], 112, 112, 3), dtype = 'float32')
    f.create_dataset('y_train', (train_list.shape[0], 112, 112, 3), dtype = 'float32')
    f.create_dataset('x_test', (test_list.shape[0], 112, 112, 3), dtype = 'float32')
    f.create_dataset('x_val', (val_list.shape[0], 112, 112, 3), dtype = 'float32')
    f.create_dataset('y_val', (val_list.shape[0], 112, 112, 3), dtype = 'float32')
    for i in range(train_list.shape[0]):
        print('0 ', i)
        f['x_train'][i] = cv2.imread(train_filepath + '/' + train_list.iloc[i,0])/255
        f['y_train'][i] = cv2.imread(train_filepath + '/' + train_list.iloc[i,1])/255
    for i in range(test_list.shape[0]):
        print('1 ', i)
        f['x_test'][i] = cv2.imread(test_filepath + '/' + test_list.iloc[i,0])/255
    for i in range(val_list.shape[0]):
        print('2 ', i)
        f['x_val'][i] = cv2.imread(val_filepath + '/' + val_list.iloc[i,0])/255
        f['y_val'][i] = cv2.imread(val_filepath + '/' + val_list.iloc[i,1])/255

with h5py.File(folder_path + '/data/data_set_32.h5', 'w') as f:
    f.create_dataset('x_train_32', (train_list.shape[0], 32, 32, 3), dtype = 'float32')
    f.create_dataset('y_train_32', (train_list.shape[0], 32, 32, 3), dtype = 'float32')
    f.create_dataset('x_test_32', (test_list.shape[0], 32, 32, 3), dtype = 'float32')
    f.create_dataset('x_val_32', (val_list.shape[0], 32, 32, 3), dtype = 'float32')
    f.create_dataset('y_val_32', (val_list.shape[0], 32, 32, 3), dtype = 'float32')
    for i in range(train_list.shape[0]):
        print('3 ', i)
        f['x_train_32'][i] = cv2.resize(cv2.imread(train_filepath + '/' + train_list.iloc[i,0]), dsize=(32, 32), interpolation=cv2.INTER_AREA)/255
        f['y_train_32'][i] = cv2.resize(cv2.imread(train_filepath + '/' + train_list.iloc[i,1]), dsize=(32, 32), interpolation=cv2.INTER_AREA)/255
    for i in range(test_list.shape[0]):
        print('4 ', i)
        f['x_test_32'][i] = cv2.resize(cv2.imread(test_filepath + '/' + test_list.iloc[i,0]), dsize=(32, 32), interpolation=cv2.INTER_AREA)/255
    for i in range(val_list.shape[0]):
        print('5 ', i)
        f['x_val_32'][i] = cv2.resize(cv2.imread(val_filepath + '/' + val_list.iloc[i,0]), dsize=(32, 32), interpolation=cv2.INTER_AREA)/255
        f['y_val_32'][i] = cv2.resize(cv2.imread(val_filepath + '/' + val_list.iloc[i,1]), dsize=(32, 32), interpolation=cv2.INTER_AREA)/255

with h5py.File(folder_path + '/data/data_set_dnn.h5', 'w') as f:
    f.create_dataset('x_train_dnn', (train_list.shape[0], 112* 112* 3), dtype = 'float32')
    f.create_dataset('y_train_dnn', (train_list.shape[0], 112* 112* 3), dtype = 'float32')
    f.create_dataset('x_test_dnn', (test_list.shape[0], 112* 112* 3), dtype = 'float32')
    f.create_dataset('x_val_dnn', (val_list.shape[0], 112* 112* 3), dtype = 'float32')
    f.create_dataset('y_val_dnn', (val_list.shape[0], 112* 112* 3), dtype = 'float32')
    for i in range(train_list.shape[0]):
        print('0 ', i)
        f['x_train_dnn'][i] = cv2.imread(train_filepath + '/' + train_list.iloc[i,0]).reshape(112*112*3)/255
        f['y_train_dnn'][i] = cv2.imread(train_filepath + '/' + train_list.iloc[i,1]).reshape(112*112*3)/255
    for i in range(test_list.shape[0]):
        print('1 ', i)
        f['x_test_dnn'][i] = cv2.imread(test_filepath + '/' + test_list.iloc[i,0]).reshape(112*112*3)/255
    for i in range(val_list.shape[0]):
        print('2 ', i)
        f['x_val_dnn'][i] = cv2.imread(val_filepath + '/' + val_list.iloc[i,0]).reshape(112*112*3)/255
        f['y_val_dnn'][i] = cv2.imread(val_filepath + '/' + val_list.iloc[i,1]).reshape(112*112*3)/255

with h5py.File(folder_path + '/data/test_data_set.h5', 'w') as f:
    f.create_dataset('x_test', (test_list.shape[0], 112, 112, 3), dtype = 'float32')
    f.create_dataset('x_test_32', (test_list.shape[0], 32, 32, 3), dtype = 'float32')
    for i in range(test_list.shape[0]):
        print('6 ', i)
        f['x_test'][i] = cv2.imread(test_filepath + '/' + test_list.iloc[i,0])/255
    for i in range(test_list.shape[0]):
        print('7 ', i)
        f['x_test_32'][i] = cv2.resize(cv2.imread(test_filepath + '/' + test_list.iloc[i,0]), dsize=(32, 32), interpolation=cv2.INTER_AREA)/255

print(train_list.shape)
print(test_list.shape)
print(val_list.shape)


#############################################
############## makemodel.py #################
#############################################

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
x_train_32 = HDF5Matrix(folder_path + '/data/data_set_32.h5', 'x_train_32', end = 100000)
y_train_32 = HDF5Matrix(folder_path + '/data/data_set_32.h5', 'y_train_32', end = 100000)
x_val_32 = HDF5Matrix(folder_path + '/data/data_set_32.h5', 'x_val_32', end = 100000)
y_val_32 = HDF5Matrix(folder_path + '/data/data_set_32.h5', 'y_val_32', end = 100000)

print(x_train.shape)    # (283336, 112, 112, 3)
print(y_train.shape)    # (283336, 112, 112, 3)
print(x_val.shape)      # (106330, 112, 112, 3)
print(y_val.shape)      # (106330, 112, 112, 3)
# print(x_train_32.shape)    # (283336, 32, 32, 3)
# print(y_train_32.shape)    # (283336, 32, 32, 3)
# print(x_val_32.shape)      # (106330, 32, 32, 3)
# print(y_val_32.shape)      # (106330, 32, 32, 3



# 2. 모델
# input_x = Input(shape = (112,112,3))
# # conv1 = AveragePooling2D((2,2))(input_x)
# conv1 = Conv2D(3, (5,5),activation='relu', padding='same')(input_x)
# conv1 = BatchNormalization()(conv1)
# # conv1 = Conv2D(8, (5,5),activation='relu', padding='same')(input_x)
# # conv1 = BatchNormalization()(conv1)
# # conv1 = Conv2D(16, (5,5),activation='relu', padding='same')(conv1)
# # conv1 = BatchNormalization()(conv1)

# # merge = concatenate([conv1, input_x])
# # conv1 = Conv2DTranspose(3, (1,1),activation='relu')(conv1)
# conv1 = Add()([conv1, input_x])
# model = Model(inputs= input_x, outputs = conv1)


input_x1 = Input(shape = (32,32,3))
conv1 = Conv2D(4, (5,5),activation='relu', padding='same')(input_x1)
conv1 = BatchNormalization()(conv1)
conv1 = Conv2D(8, (5,5),activation='relu', padding='same')(conv1)
conv1 = BatchNormalization()(conv1)
# conv1 = Conv2D(32, (5,5),activation='relu', padding='same')(conv1)
# conv1 = BatchNormalization()(conv1)

# merge = concatenate([conv1, input_y])
# conv1 = Conv2D(3, (1,1),activation='relu', padding='same')(merge)
conv1 = Conv2DTranspose(3, (1,1))(conv1)
conv1 = Add()([conv1, input_x1])
model = Model(inputs= input_x1, outputs = conv1)


model.summary()

''' 3. 훈련 '''
# earlystopping & modelcheckpoint
es = EarlyStopping(monitor='val_loss', patience=8, mode='auto')

modelpath = folder_path + '/model_check/{epoch:02d}-{val_loss:.10f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)

model.compile(loss='mae', optimizer='adam', metrics=['mae'])
hist = model.fit(x_train_32, y_train_32, epochs=40, batch_size=500, validation_data = (x_val_32,y_val_32), shuffle='batch', callbacks=[cp])

## 저장은 modelcheckpoint




#############################################
############## load_model.py ################
#############################################
import os, glob, numpy as np, cv2, zipfile, pandas as pd, shutil
from keras.models import load_model
from keras.utils.io_utils import HDF5Matrix

folder_path = os.path.dirname(os.path.realpath(__file__))

# 1. 데이터
test_list = pd.read_csv(folder_path + '/test/test_labels.csv', sep=',', header = None, index_col = None)
x_test = HDF5Matrix(folder_path + '/data/test_data_set.h5', 'x_test')

print(test_list.shape)
print(x_test.shape)

model = load_model(folder_path  + '/model_check/06-0.1940764189.hdf5')

# earlystopping & modelcheckpoint
y_pred = model.predict(x_test)

if os.path.isdir(folder_path +'/pred'):
    shutil.rmtree(folder_path +'/pred')

os.mkdir(folder_path +'/pred')

for index, img in enumerate(y_pred):
    print ('image :', index)
    cv2.imwrite(folder_path +'/pred/'+test_list.iloc[index,0], img*255)

##  터미널에서  zip pred.zip -r pred 입력시 통쨰로 압축



import os, glob, numpy as np, cv2, pandas as pd
from sklearn.model_selection import train_test_split
import h5py

folder_path = os.path.dirname(os.path.realpath(__file__))
train_filepath = folder_path + '/train'
test_filepath = folder_path + '/test'
val_filepath = folder_path + '/validate'
train_list = pd.read_csv(train_filepath + '/train_labels.csv', sep=',', header = None, index_col = None)
test_list = pd.read_csv(test_filepath + '/test_labels.csv', sep=',', header = None, index_col = None)
val_list = pd.read_csv(val_filepath + '/validate_labels.csv', sep=',', header = None, index_col = None)

print(folder_path)

image_w = 112
image_h = 112

with h5py.File(folder_path + '/data/data_set.h5', 'w') as f:
    f.create_dataset('x_train', (train_list.shape[0], 112, 112, 3), dtype = 'float32')
    f.create_dataset('y_train', (train_list.shape[0], 112, 112, 3), dtype = 'float32')
    f.create_dataset('x_test', (test_list.shape[0], 112, 112, 3), dtype = 'float32')
    f.create_dataset('x_val', (val_list.shape[0], 112, 112, 3), dtype = 'float32')
    f.create_dataset('y_val', (val_list.shape[0], 112, 112, 3), dtype = 'float32')
    for i in range(train_list.shape[0]):
        print('0 ', i)
        f['x_train'][i] = cv2.imread(train_filepath + '/' + train_list.iloc[i,0])/255
        f['y_train'][i] = cv2.imread(train_filepath + '/' + train_list.iloc[i,1])/255
    for i in range(test_list.shape[0]):
        print('1 ', i)
        f['x_test'][i] = cv2.imread(test_filepath + '/' + test_list.iloc[i,0])/255
    for i in range(val_list.shape[0]):
        print('2 ', i)
        f['x_val'][i] = cv2.imread(val_filepath + '/' + val_list.iloc[i,0])/255
        f['y_val'][i] = cv2.imread(val_filepath + '/' + val_list.iloc[i,1])/255

with h5py.File(folder_path + '/data/data_set_32.h5', 'w') as f:
    f.create_dataset('x_train_32', (train_list.shape[0], 32, 32, 3), dtype = 'float32')
    f.create_dataset('y_train_32', (train_list.shape[0], 32, 32, 3), dtype = 'float32')
    f.create_dataset('x_test_32', (test_list.shape[0], 32, 32, 3), dtype = 'float32')
    f.create_dataset('x_val_32', (val_list.shape[0], 32, 32, 3), dtype = 'float32')
    f.create_dataset('y_val_32', (val_list.shape[0], 32, 32, 3), dtype = 'float32')
    for i in range(train_list.shape[0]):
        print('3 ', i)
        f['x_train_32'][i] = cv2.resize(cv2.imread(train_filepath + '/' + train_list.iloc[i,0]), dsize=(32, 32), interpolation=cv2.INTER_AREA)/255
        f['y_train_32'][i] = cv2.resize(cv2.imread(train_filepath + '/' + train_list.iloc[i,1]), dsize=(32, 32), interpolation=cv2.INTER_AREA)/255
    for i in range(test_list.shape[0]):
        print('4 ', i)
        f['x_test_32'][i] = cv2.resize(cv2.imread(test_filepath + '/' + test_list.iloc[i,0]), dsize=(32, 32), interpolation=cv2.INTER_AREA)/255
    for i in range(val_list.shape[0]):
        print('5 ', i)
        f['x_val_32'][i] = cv2.resize(cv2.imread(val_filepath + '/' + val_list.iloc[i,0]), dsize=(32, 32), interpolation=cv2.INTER_AREA)/255
        f['y_val_32'][i] = cv2.resize(cv2.imread(val_filepath + '/' + val_list.iloc[i,1]), dsize=(32, 32), interpolation=cv2.INTER_AREA)/255

with h5py.File(folder_path + '/data/data_set_dnn.h5', 'w') as f:
    f.create_dataset('x_train_dnn', (train_list.shape[0], 112* 112* 3), dtype = 'float32')
    f.create_dataset('y_train_dnn', (train_list.shape[0], 112* 112* 3), dtype = 'float32')
    f.create_dataset('x_test_dnn', (test_list.shape[0], 112* 112* 3), dtype = 'float32')
    f.create_dataset('x_val_dnn', (val_list.shape[0], 112* 112* 3), dtype = 'float32')
    f.create_dataset('y_val_dnn', (val_list.shape[0], 112* 112* 3), dtype = 'float32')
    for i in range(train_list.shape[0]):
        print('0 ', i)
        f['x_train_dnn'][i] = cv2.imread(train_filepath + '/' + train_list.iloc[i,0]).reshape(112*112*3)/255
        f['y_train_dnn'][i] = cv2.imread(train_filepath + '/' + train_list.iloc[i,1]).reshape(112*112*3)/255
    for i in range(test_list.shape[0]):
        print('1 ', i)
        f['x_test_dnn'][i] = cv2.imread(test_filepath + '/' + test_list.iloc[i,0]).reshape(112*112*3)/255
    for i in range(val_list.shape[0]):
        print('2 ', i)
        f['x_val_dnn'][i] = cv2.imread(val_filepath + '/' + val_list.iloc[i,0]).reshape(112*112*3)/255
        f['y_val_dnn'][i] = cv2.imread(val_filepath + '/' + val_list.iloc[i,1]).reshape(112*112*3)/255

with h5py.File(folder_path + '/data/test_data_set.h5', 'w') as f:
    f.create_dataset('x_test', (test_list.shape[0], 112, 112, 3), dtype = 'float32')
    f.create_dataset('x_test_32', (test_list.shape[0], 32, 32, 3), dtype = 'float32')
    for i in range(test_list.shape[0]):
        print('6 ', i)
        f['x_test'][i] = cv2.imread(test_filepath + '/' + test_list.iloc[i,0])/255
    for i in range(test_list.shape[0]):
        print('7 ', i)
        f['x_test_32'][i] = cv2.resize(cv2.imread(test_filepath + '/' + test_list.iloc[i,0]), dsize=(32, 32), interpolation=cv2.INTER_AREA)/255

print(train_list.shape)
print(test_list.shape)
print(val_list.shape)




import os, glob, numpy as np, cv2
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Conv2DTranspose,MaxPooling2D, Dense, Flatten, Dropout, concatenate, AveragePooling2D, BatchNormalization, Add
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from keras.utils.io_utils import HDF5Matrix
from keras.initializers import RandomNormal

folder_path = os.path.dirname(os.path.realpath(__file__))

# 1. 데이터
x_train = HDF5Matrix(folder_path + '/data/data_set_dnn.h5', 'x_train_dnn')
y_train = HDF5Matrix(folder_path + '/data/data_set_dnn.h5', 'y_train_dnn')
x_val = HDF5Matrix(folder_path + '/data/data_set_dnn.h5', 'x_val_dnn')
y_val = HDF5Matrix(folder_path + '/data/data_set_dnn.h5', 'y_val_dnn')

print(x_train.shape)    # (283336, 112* 112* 3)
print(y_train.shape)    # (283336, 112* 112* 3)
print(x_val.shape)      # (106330, 112* 112* 3)
print(y_val.shape)      # (106330, 112* 112* 3)



# 2. 모델
input_x = Input(shape = (112*112*3, ))

dense1 = Dense(1000, activation='elu')(input_x)
dense1 = Dense(1000, activation='elu')(input_x)
dense1 = Dense(1000, activation='elu')(input_x)

dense1 = Dense(112*112*3, activation='elu')(input_x)



# merge = concatenate([conv1, input_x])
# conv1 = Conv2D(3, (1,1),activation='relu', padding='same')(merge)

model = Model(inputs= input_x, outputs = conv1)

model.summary()

''' 3. 훈련 '''
# earlystopping & modelcheckpoint
es = EarlyStopping(monitor='val_loss', patience=8, mode='auto')

modelpath = folder_path + '/model_check/dnn-{epoch:02d}-{val_loss:.10f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)

model.compile(loss='mae', optimizer='adam', metrics=['mae'])
hist = model.fit(x_train, y_train, epochs=40, batch_size=500, validation_data = (x_val,y_val), shuffle='batch', callbacks=[cp])

## 저장은 modelcheckpoint



import os, glob, numpy as np, cv2
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Conv2DTranspose,MaxPooling2D, Dense, Flatten, Dropout, concatenate, AveragePooling2D, BatchNormalization, Add
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from keras.utils.io_utils import HDF5Matrix
from keras.initializers import RandomNormal

folder_path = os.path.dirname(os.path.realpath(__file__))

# 1. 데이터
x_train = np.load(folder_path + '/x_train.npy')
y_train = np.load(folder_path + '/y_train.npy')
x_val = np.load(folder_path + '/x_val.npy')
y_val = np.load(folder_path + '/y_val.npy')

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
conv1 = Conv2D(3, (5,5),activation='relu', padding='same')(input_x)
conv1 = BatchNormalization()(conv1)
# conv1 = Conv2D(8, (5,5),activation='relu', padding='same')(input_x)
# conv1 = BatchNormalization()(conv1)
# conv1 = Conv2D(16, (5,5),activation='relu', padding='same')(conv1)
# conv1 = BatchNormalization()(conv1)

# merge = concatenate([conv1, input_x])
# conv1 = Conv2DTranspose(3, (1,1),activation='relu')(conv1)
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
x_train_32 = HDF5Matrix(folder_path + '/data/data_set_32.h5', 'x_train_32', end = 100000)
y_train_32 = HDF5Matrix(folder_path + '/data/data_set_32.h5', 'y_train_32', end = 100000)
x_val_32 = HDF5Matrix(folder_path + '/data/data_set_32.h5', 'x_val_32', end = 100000)
y_val_32 = HDF5Matrix(folder_path + '/data/data_set_32.h5', 'y_val_32', end = 100000)

print(x_train.shape)    # (283336, 112, 112, 3)
print(y_train.shape)    # (283336, 112, 112, 3)
print(x_val.shape)      # (106330, 112, 112, 3)
print(y_val.shape)      # (106330, 112, 112, 3)
# print(x_train_32.shape)    # (283336, 32, 32, 3)
# print(y_train_32.shape)    # (283336, 32, 32, 3)
# print(x_val_32.shape)      # (106330, 32, 32, 3)
# print(y_val_32.shape)      # (106330, 32, 32, 3



# 2. 모델
# input_x = Input(shape = (112,112,3))
# # conv1 = AveragePooling2D((2,2))(input_x)
# conv1 = Conv2D(3, (5,5),activation='relu', padding='same')(input_x)
# conv1 = BatchNormalization()(conv1)
# # conv1 = Conv2D(8, (5,5),activation='relu', padding='same')(input_x)
# # conv1 = BatchNormalization()(conv1)
# # conv1 = Conv2D(16, (5,5),activation='relu', padding='same')(conv1)
# # conv1 = BatchNormalization()(conv1)

# # merge = concatenate([conv1, input_x])
# # conv1 = Conv2DTranspose(3, (1,1),activation='relu')(conv1)
# conv1 = Add()([conv1, input_x])
# model = Model(inputs= input_x, outputs = conv1)


input_x1 = Input(shape = (32,32,3))
conv1 = Conv2D(4, (5,5),activation='relu', padding='same')(input_x1)
conv1 = BatchNormalization()(conv1)
conv1 = Conv2D(8, (5,5),activation='relu', padding='same')(conv1)
conv1 = BatchNormalization()(conv1)
# conv1 = Conv2D(32, (5,5),activation='relu', padding='same')(conv1)
# conv1 = BatchNormalization()(conv1)

# merge = concatenate([conv1, input_y])
# conv1 = Conv2D(3, (1,1),activation='relu', padding='same')(merge)
conv1 = Conv2DTranspose(3, (1,1))(conv1)
conv1 = Add()([conv1, input_x1])
model = Model(inputs= input_x1, outputs = conv1)


model.summary()

''' 3. 훈련 '''
# earlystopping & modelcheckpoint
es = EarlyStopping(monitor='val_loss', patience=8, mode='auto')

modelpath = folder_path + '/model_check/{epoch:02d}-{val_loss:.10f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)

model.compile(loss='mae', optimizer='adam', metrics=['mae'])
hist = model.fit(x_train_32, y_train_32, epochs=40, batch_size=500, validation_data = (x_val_32,y_val_32), shuffle='batch', callbacks=[cp])

## 저장은 modelcheckpoint



import os, glob, numpy as np, cv2, zipfile, pandas as pd, shutil
from keras.models import load_model
from keras.utils.io_utils import HDF5Matrix

folder_path = os.path.dirname(os.path.realpath(__file__))

# 1. 데이터
test_list = pd.read_csv(folder_path + '/test/test_labels.csv', sep=',', header = None, index_col = None)
x_test = HDF5Matrix(folder_path + '/data/test_data_set.h5', 'x_test')

print(test_list.shape)
print(x_test.shape)

model = load_model(folder_path  + '/model_check/06-0.1940764189.hdf5')

# earlystopping & modelcheckpoint
y_pred = model.predict(x_test)

if os.path.isdir(folder_path +'/pred'):
    shutil.rmtree(folder_path +'/pred')

os.mkdir(folder_path +'/pred')

for index, img in enumerate(y_pred):
    print ('image :', index)
    cv2.imwrite(folder_path +'/pred/'+test_list.iloc[index,0], img*255)

##  터미널에서  zip pred.zip -r pred 입력시 통쨰로 압축



import os, glob, numpy as np, cv2, zipfile, pandas as pd, shutil
from keras.models import load_model
from keras.utils.io_utils import HDF5Matrix

folder_path = os.path.dirname(os.path.realpath(__file__))

# 1. 데이터
test_list = pd.read_csv(folder_path + '/test/test_labels.csv', sep=',', header = None, index_col = None)
x_test_32 = HDF5Matrix(folder_path + '/data/test_data_set.h5', 'x_test_32')

print(test_list.shape)
print(x_test_32.shape)

model = load_model(folder_path  + '/model_check/06-0.1940764189.hdf5')

# earlystopping & modelcheckpoint
y_pred = model.predict(x_test_32)

if os.path.isdir(folder_path +'/pred'):
    shutil.rmtree(folder_path +'/pred')

os.mkdir(folder_path +'/pred')

for index, img in enumerate(y_pred):
    print ('image :', index)
    cv2.imwrite(folder_path +'/pred/'+test_list.iloc[index,0], cv2.resize(img*255, dsize=(112,112), interpolation=cv2.INTER_LINEAR))

##  터미널에서  zip pred.zip -r pred 입력시 통쨰로 압축
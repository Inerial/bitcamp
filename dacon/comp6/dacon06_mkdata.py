import os, glob, numpy as np, pandas as pd
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
import h5py

data_path = './data/dacon/comp6'
train_path = data_path + '/train'
test_path = data_path + '/test'
train_list = os.listdir(train_path)
test_list = os.listdir(test_path)

y_train = pd.read_csv(data_path+'/train_answer.csv', sep=',', index_col=0, header=0).values

x_train = []
for train_data in train_list:
    fs, data = wavfile.read(train_path + '/' + train_data)
    Y_real = np.fft.fft(data, n = 32000).real
    Y_imag = np.fft.fft(data, n = 32000).imag
    P_real = abs(Y_real/fs)[:16000]
    P_imag = abs(Y_imag/fs)[:16000]
    x_train.append(np.transpose([data, P_real, P_imag]))
x_train = np.array(x_train)

x_test = []
for test_data in test_list:
    fs, data = wavfile.read(test_path + '/' + test_data)
    Y_real = np.fft.fft(data, n = 32000).real
    Y_imag = np.fft.fft(data, n = 32000).imag
    P_real = abs(Y_real/fs)[:16000]
    P_imag = abs(Y_imag/fs)[:16000]
    x_test.append(np.transpose([data, P_real, P_imag]))
x_test = np.array(x_test)


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)


''' 
with h5py.File(data_path + '/data_set.h5', 'w') as f:
    f.create_dataset('x_train', (train_list.shape[0], 112, 112, 3), dtype = 'float32')
    f.create_dataset('y_train', (train_list.shape[0], 112, 112, 3), dtype = 'float32')
    f.create_dataset('x_test', (test_list.shape[0], 112, 112, 3), dtype = 'float32')
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

# with h5py.File(folder_path + '/data/data_set_32.h5', 'w') as f:
#     f.create_dataset('x_train_32', (train_list.shape[0], 32, 32, 3), dtype = 'float32')
#     f.create_dataset('y_train_32', (train_list.shape[0], 32, 32, 3), dtype = 'float32')
#     f.create_dataset('x_test_32', (test_list.shape[0], 32, 32, 3), dtype = 'float32')
#     f.create_dataset('x_val_32', (val_list.shape[0], 32, 32, 3), dtype = 'float32')
#     f.create_dataset('y_val_32', (val_list.shape[0], 32, 32, 3), dtype = 'float32')
#     for i in range(train_list.shape[0]):
#         print('3 ', i)
#         f['x_train_32'][i] = cv2.resize(cv2.imread(train_filepath + '/' + train_list.iloc[i,0]), dsize=(32, 32), interpolation=cv2.INTER_AREA)/255
#         f['y_train_32'][i] = cv2.resize(cv2.imread(train_filepath + '/' + train_list.iloc[i,1]), dsize=(32, 32), interpolation=cv2.INTER_AREA)/255
#     for i in range(test_list.shape[0]):
#         print('4 ', i)
#         f['x_test_32'][i] = cv2.resize(cv2.imread(test_filepath + '/' + test_list.iloc[i,0]), dsize=(32, 32), interpolation=cv2.INTER_AREA)/255
#     for i in range(val_list.shape[0]):
#         print('5 ', i)
#         f['x_val_32'][i] = cv2.resize(cv2.imread(val_filepath + '/' + val_list.iloc[i,0]), dsize=(32, 32), interpolation=cv2.INTER_AREA)/255
#         f['y_val_32'][i] = cv2.resize(cv2.imread(val_filepath + '/' + val_list.iloc[i,1]), dsize=(32, 32), interpolation=cv2.INTER_AREA)/255

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

# with h5py.File(folder_path + '/data/test_data_set.h5', 'w') as f:
#     f.create_dataset('x_test', (test_list.shape[0], 112, 112, 3), dtype = 'float32')
#     f.create_dataset('x_test_32', (test_list.shape[0], 32, 32, 3), dtype = 'float32')
#     for i in range(test_list.shape[0]):
#         print('6 ', i)
#         f['x_test'][i] = cv2.imread(test_filepath + '/' + test_list.iloc[i,0])/255
#     for i in range(test_list.shape[0]):
#         print('7 ', i)
#         f['x_test_32'][i] = cv2.resize(cv2.imread(test_filepath + '/' + test_list.iloc[i,0]), dsize=(32, 32), interpolation=cv2.INTER_AREA)/255

print(train_list.shape)
print(test_list.shape)
print(val_list.shape) '''
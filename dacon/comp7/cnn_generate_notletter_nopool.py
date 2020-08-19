import numpy as np, cv2
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization,ReLU, LeakyReLU, ELU,concatenate
from tensorflow.keras.activations import selu
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam,Adagrad,Adamax,RMSprop,Nadam
import math
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1, l2, l1_l2

class CosineAnnealingScheduler(Callback):
    """Cosine annealing scheduler.
    """
    def __init__(self, T_max, eta_max, eta_min=0, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        

train = pd.read_csv('./data/dacon/comp7/train.csv', sep=',', header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp7/test.csv', sep=',', header = 0, index_col = 0)
submit = pd.read_csv('./data/dacon/comp7/submission.csv', sep=',', header = 0, index_col = 0)

# x_train_letter = train.values[:,1]
# train.iloc[:,1] = np.array([ord(i)-ord('A')+1 for i in x_train_letter])

# x_test_letter = test.values[:,0]
# test.iloc[:,0] = np.array([ord(i)-ord('A')+1 for i in x_test_letter])

from keras.preprocessing.image import ImageDataGenerator
gen = ImageDataGenerator(
        rotation_range=10, 
        # shear_range= 0.10,
        zoom_range = 0.10,  
        width_shift_range=0.1, 
        height_shift_range=0.1,
        validation_split = 0.1,
        )  # randomly flip images

x_train = train.values[:, 2:]
x_train = x_train.reshape(-1, 28,28).astype(int)
y_train = train.values[:,0] ## 숫자 
x_train_letter = train.values[:,1]
x_train_letter = np.array([ord(i)-ord('A') for i in x_train_letter])
x_train_letter = to_categorical(x_train_letter)
y_train = to_categorical(y_train)

x_train_channel = []
for i in range(len(x_train)):
    mkchannel = []
    tmp = x_train[i]*2
    tmp[tmp>255] = 255
    for j in range(26):
        if int(x_train_letter[i,j]) is 0:
            # mkchannel.append(x_train[i]/25)
            # mkchannel.append(np.zeros(x_train[i].shape))
            mkchannel.append(np.zeros(x_train[i].shape))
        else:
            mkchannel.append(x_train[i])
            # mkchannel.append(tmp)
    mkchannel = np.array(mkchannel)
    x_train_channel.append(mkchannel.reshape(112,182,1))

x_train = np.array(x_train_channel)/255.
print(x_train.shape)
# x_train, x_val, y_train, y_val = train_test_split(
#     x_train,y_train, train_size=0.9, shuffle=True, random_state=66
# )

gen.fit(x_train)

x_real = test.values[:, 1:]
x_real = x_real.reshape(-1, 28,28).astype(int)
x_real_letter = test.values[:,0]
x_real_letter = np.array([ord(i)-ord('A') for i in x_real_letter])
x_real_letter = to_categorical(x_real_letter)

x_real_channel = []
for i in range(len(x_real)):
    mkchannel1 = []
    tmp = x_real[i]*2
    tmp[tmp>255] = 255
    for j in range(26):
        if int(x_real_letter[i,j]) is 0:
            # mkchannel1.append(x_real[i]/25)
            mkchannel1.append(np.zeros(x_real[i].shape))
            # mkchannel1.append(np.zeros(x_real[i].shape))
        else:
            mkchannel1.append(x_real[i])
            # mkchannel1.append(tmp)
    
    mkchannel1 = np.array(mkchannel1)
    x_real_channel.append(mkchannel1.reshape(112,182,1))

x_real = np.array(x_real_channel)/255.
print(x_real.shape)






########################################
################ 모델링 #################
########################################
regul = None#l1(0.002)

input1 = Input(shape=(112,182,1))
conv1 = BatchNormalization()(input1)

conv1 = Conv2D(64,(3,3), kernel_regularizer=regul)(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = ReLU()(conv1)
conv1 = Conv2D(64,(3,3), kernel_regularizer=regul)(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = ReLU()(conv1)
conv1 = Conv2D(64,(5,5),strides=2,padding='same', kernel_regularizer=regul)(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = ReLU()(conv1)
# conv1 = Dropout(0.4)(conv1)

conv1 = Conv2D(128,(3,3), kernel_regularizer=regul)(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = ReLU()(conv1)
conv1 = Conv2D(128,(3,3), kernel_regularizer=regul)(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = ReLU()(conv1)
conv1 = Conv2D(128,(5,5),strides=2,padding='same', kernel_regularizer=regul)(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = ReLU()(conv1)
# conv1 = Dropout(0.4)(conv1)

conv1 = Conv2D(256,(4,4), kernel_regularizer=regul)(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = ReLU()(conv1)

conv1 = Flatten()(conv1)
# conv1 = Dropout(0.4)(conv1)
conv1 = BatchNormalization()(conv1)

conv1 = Dense(10, activation='softmax')(conv1)

model = Model(inputs=input1, outputs= conv1)
model.summary()

# optimizers = Adam(epsilon=1e-08)
optimizers = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizers, loss='categorical_crossentropy', metrics=['acc'])

reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
# reduction = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
# reduction = CosineAnnealingScheduler(T_max=300, eta_max=1e-3, eta_min=0.00001, verbose=1)

check = ModelCheckpoint('./dacon/comp7/bestcheck.hdf5', monitor='val_acc',save_best_only=True)


batch_size = 32
epoch = 100
print(x_train.shape)
model.fit_generator(gen.flow(x_train, y_train,batch_size=batch_size, subset='training'),
                    steps_per_epoch=int(x_train.shape[0]/batch_size), epochs=epoch,
                    validation_data=gen.flow(x_train, y_train,batch_size=batch_size, subset='validation'),
                    callbacks=[check, reduction], validation_steps=int(x_train.shape[0]*0.1/batch_size))

model = load_model('./dacon/comp7/bestcheck.hdf5')
y_pred = model.predict(x_real)

submit['digit'] = np.argmax(y_pred,axis=1)
print(submit)
submit.to_csv('./dacon/comp7/comp7_sub.csv')
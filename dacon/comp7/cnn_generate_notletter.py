import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization,ReLU, LeakyReLU, ELU,concatenate
from keras.activations import selu
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, LearningRateScheduler
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.optimizers import Adam,Adagrad,Adamax,RMSprop,Nadam
import math
from keras.callbacks import Callback
from keras import backend as K
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

from keras.preprocessing.image import ImageDataGenerator
gen = ImageDataGenerator(
        rotation_range=30, 
        zoom_range = 0.10,  
        width_shift_range=0.1, 
        height_shift_range=0.1,
        validation_split = 0.2,

        )  # randomly flip images

x_train = train.values[:, 2:].reshape(-1, 28,28,1)/255
# x_train[x_train < 0.2] = 2
# x_train[x_train < 0.6] = 0
# x_train[x_train == 2] = 0.2

y_train = train.values[:,0] ## 숫자 

x_train_letter = train.values[:,1]
x_train_letter = np.array([ord(i)-ord('A') for i in x_train_letter])
x_train_letter = to_categorical(x_train_letter)

y_train = to_categorical(y_train)
# y.append(train.values[i,1]) ## 알파벳
    
# x_train, x_val, y_train, y_val = train_test_split(
#     x_train,y_train, train_size=0.9, shuffle=True, random_state=66
# )

gen.fit(x_train)

x_real = test.values[:, 1:].reshape(-1, 28,28,1)/255
# x_real[x_real < 0.2] = 2
# x_real[x_real < 0.60] = 0
# x_real[x_real == 2] = 0.2
x_real_letter = test.values[:,0]
x_real_letter = np.array([ord(i)-ord('A') for i in x_real_letter])
x_real_letter = to_categorical(x_real_letter)

print(x_real.shape)






########################################
################ 모델링 #################
########################################
regul = None#l1(0.002)

input1 = Input(shape=(28,28,1))
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

# optimizers = Nadam(epsilon=1e-08)
optimizers = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizers, loss='categorical_crossentropy', metrics=['acc'])

reduction = ReduceLROnPlateau(monitor='val_acc', patience=5, verbose=1, factor=0.5, min_lr=0.000001)
# reduction = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
# reduction = CosineAnnealingScheduler(T_max=300, eta_max=1e-3, eta_min=0.00001, verbose=1)

check = ModelCheckpoint('./dacon/comp7/bestcheck.hdf5', monitor='val_acc',save_best_only=True)


batch_size = 32
epoch = 500
model.fit_generator(gen.flow(x_train, y_train,batch_size=batch_size,subset='training'),
                    steps_per_epoch=int(x_train.shape[0]*0.8/batch_size), epochs=epoch,
                    validation_data=gen.flow(x_train, y_train,batch_size=batch_size,subset='validation'),
                    callbacks=[check, reduction], validation_steps=int(x_train.shape[0]*0.2/batch_size))

model = load_model('./dacon/comp7/bestcheck.hdf5')
y_pred = model.predict(x_real)

submit['digit'] = np.argmax(y_pred,axis=1)
print(submit)
submit.to_csv('./dacon/comp7/comp7_sub.csv')
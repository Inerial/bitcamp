import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization,ReLU
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.optimizers import Adam,Adagrad,Adamax,RMSprop
train = pd.read_csv('./data/dacon/comp7/train.csv', sep=',', header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp7/test.csv', sep=',', header = 0, index_col = 0)
submit = pd.read_csv('./data/dacon/comp7/submission.csv', sep=',', header = 0, index_col = 0)

from keras.preprocessing.image import ImageDataGenerator
gen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

x_train = train.values[:, 2:].reshape(-1, 28,28,1)/255
x_train[x_train < 0.2] = 2
x_train[x_train < 0.6] = 0
x_train[x_train == 2] = 0.2

y_train = train.values[:,0] ## 숫자 


# for i in range(2048):
#     print(train.values[i,0], train.values[i,1])
#     plt.imshow((x_train[i]*255).reshape(28,28).astype(int))
#     plt.show()
y_train = np_utils.to_categorical(y_train)
# y.append(train.values[i,1]) ## 알파벳
    
x_train, x_val, y_train, y_val = train_test_split(
    x_train,y_train, train_size=0.9, shuffle=True, random_state=66
)

gen.fit(x_train)

x_real = test.values[:, 1:].reshape(-1, 28,28,1)/255
x_real[x_real < 0.2] = 2
x_real[x_real < 0.60] = 0
x_real[x_real == 2] = 0.2

print(x_real.shape)
input1 = Input(shape=(28,28,1))
conv1 = Conv2D(32,(3,3),padding='same')(input1)
conv1 = BatchNormalization()(conv1)
conv1 = ReLU()(conv1)
conv1 = Conv2D(32,(3,3),padding='same')(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = ReLU()(conv1)
conv1 = Conv2D(32,(3,3),padding='same')(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = ReLU()(conv1)

conv1 = MaxPooling2D((2,2))(conv1)
conv1 = Dropout(0.25)(conv1)

conv1 = Conv2D(64,(3,3),padding='same')(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = ReLU()(conv1)
conv1 = Conv2D(64,(3,3),padding='same')(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = ReLU()(conv1)
conv1 = Conv2D(64,(3,3),padding='same')(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = ReLU()(conv1)

conv1 = MaxPooling2D((2,2))(conv1)
conv1 = Dropout(0.25)(conv1)

conv1 = Conv2D(128,(3,3),padding='same')(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = ReLU()(conv1)
conv1 = Conv2D(128,(3,3),padding='same')(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = ReLU()(conv1)
conv1 = Conv2D(128,(3,3),padding='same')(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = ReLU()(conv1)

conv1 = MaxPooling2D((2,2))(conv1)
conv1 = Dropout(0.25)(conv1)

conv1 = Conv2D(256,(3,3),padding='same')(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = ReLU()(conv1)
conv1 = Conv2D(256,(3,3),padding='same')(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = ReLU()(conv1)
conv1 = Conv2D(256,(3,3),padding='same')(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = ReLU()(conv1)


conv1 = Flatten()(conv1)
conv1 = Dense(256, activation='relu')(conv1)
conv1 = Dropout(0.5)(conv1)
conv1 = Dense(10, activation='softmax')(conv1)

model = Model(inputs=input1, outputs= conv1)
model.summary()

model.compile(optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics=['acc'])
reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
check = ModelCheckpoint('./dacon/comp7/bestcheck.hdf5', monitor='val_acc',save_best_only=True)

model.fit_generator(gen.flow(x_train,y_train,batch_size=64), steps_per_epoch=len(x_train)/64, epochs=100,
                    validation_data=(x_val,y_val), callbacks=[check, reduction])

model = load_model('./dacon/comp7/bestcheck.hdf5')
y_pred = model.predict(x_real)

submit['digit'] = np.argmax(y_pred,axis=1)
print(submit)
submit.to_csv('./dacon/comp7/comp7_sub.csv')
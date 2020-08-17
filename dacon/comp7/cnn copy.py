import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization,ReLU, Concatenate, LeakyReLU, RepeatVector, Multiply
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.optimizers import Adam, RMSprop
train = pd.read_csv('./data/dacon/comp7/train.csv', sep=',', header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp7/test.csv', sep=',', header = 0, index_col = 0)
submit = pd.read_csv('./data/dacon/comp7/submission.csv', sep=',', header = 0, index_col = 0)

x_train = train.values[:, 2:].reshape(-1, 28,28,1)/255
x_train[x_train < 0.2] = 2
x_train[x_train < 0.6] = 0
x_train[x_train == 2] = 0.2

y_train = train.values[:,0] ## 숫자 
x_train_let = train.values[:,1] ## 문자

x_train_let = np.array([ord(i)-ord('A') for i in x_train_let])
x_train_let = to_categorical(x_train_let)

# for i in range(2048):
#     print(train.values[i,0], train.values[i,1])
#     plt.imshow((x_train[i]*255).reshape(28,28).astype(int))
#     plt.show()
y_train = to_categorical(y_train)
# y.append(train.values[i,1]) ## 알파벳
    
x_train, x_test,x_train_let, x_test_let, y_train, y_test = train_test_split(
    x_train, x_train_let,y_train, train_size=0.9, shuffle=True, random_state=66
)

x_real = test.values[:, 1:].reshape(-1, 28,28,1)/255
x_real[x_real < 0.2] = 2
x_real[x_real < 0.60] = 0
x_real[x_real == 2] = 0.2
x_real_let = test.values[:,0] ## 문자

x_real_let = np.array([ord(i)-ord('A') for i in x_real_let])
x_real_let = to_categorical(x_real_let)


print(x_real.shape)


########################################
################ 모델링 #################
########################################
input1 = Input(shape=(28,28,1))
input2 = Input(shape=(26,))

conv1 = Conv2D(64,(3,3))(input1)
# conv1 = BatchNormalization()(conv1)
conv1 = LeakyReLU()(conv1)
conv1 = Conv2D(64,(3,3))(conv1)
# conv1 = BatchNormalization()(conv1)
conv1 = LeakyReLU()(conv1)
conv1 = Conv2D(64,(5,5),strides=2,padding='same')(conv1)
# conv1 = BatchNormalization()(conv1)
conv1 = LeakyReLU()(conv1)
# conv1 = Dropout(0.25)(conv1)
conv1 = BatchNormalization()(conv1)

conv1 = Conv2D(128,(3,3))(conv1)
# conv1 = BatchNormalization()(conv1)
conv1 = LeakyReLU()(conv1)
conv1 = Conv2D(128,(3,3))(conv1)
# conv1 = BatchNormalization()(conv1)
conv1 = LeakyReLU()(conv1)
conv1 = Conv2D(128,(5,5),strides=2,padding='same')(conv1)
# conv1 = BatchNormalization()(conv1)
conv1 = LeakyReLU()(conv1)
# conv1 = Dropout(0.25)(conv1)
conv1 = BatchNormalization()(conv1)

conv1 = Conv2D(256,(4,4))(conv1)
# conv1 = BatchNormalization()(conv1)
conv1 = LeakyReLU()(conv1)

conv1 = Flatten()(conv1)
conv1 = RepeatVector(26)(conv1)
conv1 = Flatten()(conv1)
# conv1 = Dropout(0.25)(conv1)
# conv1 = BatchNormalization()(conv1)

conv2 = RepeatVector(256)(input2)
conv2 = Flatten()(conv2)

outputs = Multiply()([conv1, conv2])
outputs = ReLU()(outputs)
conv1 = Dropout(0.25)(conv1)

outputs = Dense(256, activation='relu')(outputs)
conv1 = Dropout(0.25)(conv1)
outputs = Dense(10, activation='softmax')(outputs)

model = Model(inputs=[input1, input2], outputs= outputs)
model.summary()

model.compile(optimizer= RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics=['acc'])

check = ModelCheckpoint('./dacon/comp7/bestcheck.hdf5', monitor='val_loss',save_best_only=True)
model.fit([x_train, x_train_let],y_train,batch_size=64, epochs=100, validation_split=0.2, callbacks=[check])

model = load_model('./dacon/comp7/bestcheck.hdf5')
print(model.evaluate([x_test,x_test_let],y_test))
y_pred = model.predict([x_real,x_real_let])

submit['digit'] = np.argmax(y_pred,axis=1)
print(submit)
submit.to_csv('./dacon/comp7/comp7_sub.csv')
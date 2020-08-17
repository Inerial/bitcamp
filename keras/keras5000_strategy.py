import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from keras.utils import np_utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
from sklearn.model_selection import train_test_split

strategy = tf.distribute.Strategy(extended=tf.distribute.MirroredStrategy())
(x_train, y_train),(x_test, y_test) = mnist.load_data()


## OneHotEncoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

## 정규화
x_train = x_train.reshape(60000,28,28,1).astype('float32') / 255
x_test = x_test.reshape(10000,28,28,1).astype('float32') / 255

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size = 0.2, shuffle=True, random_state=66
)

## 모델링

batch_for_st = 256
batch_size = batch_for_st * strategy.extended.num_replicas_in_sync
BUFFER_SIZE = 10000

train = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(BUFFER_SIZE).batch(batch_size)
val = tf.data.Dataset.from_tensor_slices((x_val,y_val)).shuffle(BUFFER_SIZE).batch(batch_size)
test = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(batch_size)

with strategy.extended.scope():
    input1 = Input(shape = (28,28,1))
    Conv1 = Conv2D(64, (3,3),activation='relu', input_shape=(28,28,1))(input1)
    Conv1 = Conv2D(64, (3,3),activation='relu')(Conv1)
    Conv1 = Conv2D(64, (5,5),strides=2,padding = 'same',activation='relu')(Conv1)
    Conv1 = BatchNormalization()(Conv1)
    # Conv1 = Dropout(0.4)(Conv1)


    Conv1 = Conv2D(128, (3,3),activation='relu')(Conv1)
    Conv1 = Conv2D(128, (3,3),activation='relu')(Conv1)
    Conv1 = Conv2D(128, (5,5),strides=2,padding = 'same',activation='relu')(Conv1)
    Conv1 = BatchNormalization()(Conv1)
    # Conv1 = Dropout(0.4)(Conv1)

    Conv1 = Conv2D(256, (4,4),activation='relu')(Conv1)
    Conv1 = BatchNormalization()(Conv1)
    # Conv1 = Dropout(0.5)(Conv1)

    Conv1 = Flatten()(Conv1)

    Dense1 = Dense(10, activation='softmax')(Conv1)

    model = Model(inputs = input1, outputs = Dense1)

    optimizers = RMSprop(lr=0.002, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizers, loss='categorical_crossentropy', metrics=['acc'])

reduction = ReduceLROnPlateau(monitor='loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
check = ModelCheckpoint('./bestcheck.hdf5', monitor='loss',save_best_only=True)

model.fit(train, epochs=60, validation_data=val)#, callbacks=[reduction, check])

# model.load_weights('./bestcheck.hdf5')

loss, acc = model.evaluate(test)
print('loss :',loss)
print('acc :',acc)

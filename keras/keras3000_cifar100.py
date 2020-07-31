from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.layers import Conv2D, Dense, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Input, LSTM, UpSampling2D, Conv2DTranspose
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from keras.applications import ResNet50
import tensorflow as tf
 
 
 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



def extract_resnet(X,y): 
    resnet_model = ResNet50(input_shape=(X.shape[1], X.shape[2], 3), include_top=False) 
    print(resnet_model.predict(X))
    x = Flatten()(resnet_model.layers[-1].output)
    x = Dense(10, activation='softmax')(x)
    
    model = Model(resnet_model.input, x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit(X,y,epochs=1,batch_size=64)
    features_array = resnet_model.predict(X)
    print(features_array)
    return features_array


(X_train, y_train),(X_test,y_test) = cifar10.load_data()

y_train_res = np_utils.to_categorical(y_train)
y_test_res = np_utils.to_categorical(y_test)

X_train = np.squeeze(extract_resnet(X_train, y_train_res))
X_test = np.squeeze(extract_resnet(X_test, y_test_res))


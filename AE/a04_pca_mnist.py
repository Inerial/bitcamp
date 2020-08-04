## acc : 0.982  이상 띄우기

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, UpSampling2D, Conv2DTranspose, Input

from keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train[0])
print('y_train :',y_train[0])

## OneHotEncoding
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)

## 정규화
# plt.imshow(x_train[0], 'gray')
# plt.show()
x_train = x_train.reshape(60000,784).astype('float32') / 255
x_test = x_test.reshape(10000,784).astype('float32') / 255

# np.append(x)(x_train,x_test)

from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(x_train)
cumsum=np.cumsum(pca.explained_variance_ratio_)
print(cumsum)
best_n_components = np.argmax(cumsum >= 0.99) + 1
print(best_n_components)
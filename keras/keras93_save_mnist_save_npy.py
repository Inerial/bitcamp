import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

## 1 : 데이터 전처리
(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train[0])
print('y_train :',y_train[0])

#plt.imshow(x_train[1], 'gray')
#plt.show()

#from sklearn.preprocessing import OneHotEncoder
#hot = OneHotEncoder()

## OneHotEncoding
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

## 정규화
#from sklearn.preprocessing import MinMaxScaler
x_train = x_train.reshape(60000,28,28,1).astype('float32') / 255
x_test = x_test.reshape(10000,28,28,1).astype('float32') / 255


np.save('./data/mnist_train_x.npy',arr = x_train)
np.save('./data/mnist_train_y.npy',arr = y_train)
np.save('./data/mnist_test_x.npy',arr = x_test)
np.save('./data/mnist_test_y.npy',arr = y_test)

## numpy는 한가지 자료형만 사용가능! 두개이상 불가능
## => 두개이상은 pandas를 사용해야함!
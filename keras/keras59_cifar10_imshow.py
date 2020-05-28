from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten, Input, LSTM
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train),(x_test,y_test) = cifar10.load_data()

print(x_train)
print(x_test)
print(y_train)
print(y_test)


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

plt.imshow(x_train[0])
plt.show()
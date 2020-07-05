import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
x = iris.data[:, (2,3)]
y = (iris.target == 0).astype(np.int)

per_clf = Perceptron()
per_clf.fit(x,y)

y_pred = per_clf.predict([[2, 0.5]])

print(y_pred) # [0]

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import Sequential, Model, load_model

print(tf.__version__)
print(keras.__version__)
# 2.0.0
# 2.2.4-tf

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test,y_test) = fashion_mnist.load_data()

print(x_train.shape)
print(x_train.dtype)
# (60000, 28, 28)
# uint8


x_val, x_train = x_train[:5000] / 255., x_train[5000:] / 255.
y_val, y_train = y_train[:5000], y_train[5000:]
x_test = x_test / 255.

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
class_names[y_train[0]]


model = Sequential()
model.add(Flatten(input_shape=[28, 28]))
model.add(Dense(300, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(10, activation="softmax"))



model = Sequential([
    Flatten(input_shape=[28, 28]),
    Dense(300, activation="relu"),
    Dense(100, activation="relu"),
    Dense(10, activation="softmax")
])
## 냅다 플래튼 박으면 이미지를 펼칠수 있다.



model.summary()
# Model: "sequential_1"
# Layer (type)                 Output Shape              Param #
# =================================================================
# flatten_1 (Flatten)          (None, 784)               0
# _________________________________________________________________
# dense_3 (Dense)              (None, 300)               235500
# _________________________________________________________________
# dense_4 (Dense)              (None, 100)               30100
# _________________________________________________________________
# dense_5 (Dense)              (None, 10)                1010
# =================================================================
# Total params: 266,610
# Trainable params: 266,610
# Non-trainable params: 0
# _________________________________________________________________
# >>>


print(model.layers)
# [<tensorflow.python.keras.layers.core.Flatten object at 0x00000208F4F8FEC8>, <tensorflow.python.keras.layers.core.Dense object at 0x00000208F4F920C8>, <tensorflow.python.keras.layers.core.Dense object at 0x00000208F4F92688>, 
# <tensorflow.python.keras.layers.core.Dense object at 0x00000208F4F92C88>]



hidden1 = model.layers[1]
print(hidden1.name)
# dense_3

model.get_layer(hidden1.name) is hidden1
# True

weights, biases = hidden1.get_weights()
print(weights)
# [[-0.05304039 -0.04812606  0.06779003 ... -0.01093299 -0.07131267
#    0.05638868]
#  [-0.03920617 -0.0027833   0.05087268 ... -0.01691134 -0.02497154
#   -0.00371246]
#  [-0.02350533 -0.01308121 -0.02270878 ... -0.02880067 -0.06623901
#    0.07161343]
#  ...
#  [-0.05289905  0.07030329 -0.05734645 ...  0.05359069 -0.04493336
#   -0.058148  ]
#  [-0.03356527 -0.07253792  0.04117389 ...  0.06415325 -0.00891226
#    0.02614703]
#  [-0.01270749 -0.05905373  0.05872917 ...  0.00281982  0.02470104
#    0.02896889]]
# >>>
print(weights.shape)
# (784, 300)
print(biases)
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

print(biases.shape)
# (300,)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=30,
                    validation_data=(x_val, y_val))
# Train on 55000 samples, validate on 5000 samples
# Epoch 1/30
# 55000/55000 [==============================] - 4s 69us/sample - loss: 0.7079 - accuracy: 0.7689 - val_loss: 0.5094 - val_accuracy: 0.8280
# Epoch 2/30
# 55000/55000 [==============================] - 3s 60us/sample - loss: 0.4867 - accuracy: 0.8316 - val_loss: 0.4500 - val_accuracy: 0.8526
# Epoch 3/30
# 55000/55000 [==============================] - 3s 55us/sample - loss: 0.4413 - accuracy: 0.8452 - val_loss: 0.4171 - val_accuracy: 0.8582
# Epoch 4/30
# 55000/55000 [==============================] - 3s 56us/sample - loss: 0.4131 - accuracy: 0.8551 - val_loss: 0.3987 - val_accuracy: 0.8650
# Epoch 5/30
# 55000/55000 [==============================] - 3s 64us/sample - loss: 0.3921 - accuracy: 0.8626 - val_loss: 0.4071 - val_accuracy: 0.8544
# Epoch 6/30
# 55000/55000 [==============================] - 3s 58us/sample - loss: 0.3778 - accuracy: 0.8665 - val_loss: 0.3793 - val_accuracy: 0.8670
# Epoch 7/30
# 55000/55000 [==============================] - 3s 57us/sample - loss: 0.3639 - accuracy: 0.8716 - val_loss: 0.4189 - val_accuracy: 0.8474
# Epoch 8/30
# 55000/55000 [==============================] - 3s 55us/sample - loss: 0.3511 - accuracy: 0.8742 - val_loss: 0.3664 - val_accuracy: 0.8716
# Epoch 9/30
# 55000/55000 [==============================] - 3s 59us/sample - loss: 0.3428 - accuracy: 0.8785 - val_loss: 0.3489 - val_accuracy: 0.8782
# Epoch 10/30
# 55000/55000 [==============================] - 3s 55us/sample - loss: 0.3336 - accuracy: 0.8814 - val_loss: 0.3553 - val_accuracy: 0.8752
# Epoch 11/30
# 55000/55000 [==============================] - 3s 58us/sample - loss: 0.3255 - accuracy: 0.8831 - val_loss: 0.3570 - val_accuracy: 0.8742
# Epoch 12/30
# 55000/55000 [==============================] - 3s 55us/sample - loss: 0.3178 - accuracy: 0.8850 - val_loss: 0.3490 - val_accuracy: 0.8798
# Epoch 13/30
# 55000/55000 [==============================] - 3s 61us/sample - loss: 0.3110 - accuracy: 0.8880 - val_loss: 0.3337 - val_accuracy: 0.8802
# Epoch 14/30
# 55000/55000 [==============================] - 3s 61us/sample - loss: 0.3035 - accuracy: 0.8907 - val_loss: 0.3236 - val_accuracy: 0.8836
# Epoch 15/30
# 55000/55000 [==============================] - 3s 61us/sample - loss: 0.2967 - accuracy: 0.8924 - val_loss: 0.3269 - val_accuracy: 0.8846
# Epoch 16/30
# 55000/55000 [==============================] - 3s 56us/sample - loss: 0.2915 - accuracy: 0.8939 - val_loss: 0.3212 - val_accuracy: 0.8896
# Epoch 17/30
# 55000/55000 [==============================] - 3s 59us/sample - loss: 0.2854 - accuracy: 0.8967 - val_loss: 0.3135 - val_accuracy: 0.8888
# Epoch 18/30
# 55000/55000 [==============================] - 3s 58us/sample - loss: 0.2802 - accuracy: 0.8980 - val_loss: 0.3172 - val_accuracy: 0.8856
# Epoch 19/30
# 55000/55000 [==============================] - 3s 55us/sample - loss: 0.2746 - accuracy: 0.9002 - val_loss: 0.3132 - val_accuracy: 0.8892
# Epoch 20/30
# 55000/55000 [==============================] - 3s 54us/sample - loss: 0.2693 - accuracy: 0.9035 - val_loss: 0.3170 - val_accuracy: 0.8854
# Epoch 21/30
# 55000/55000 [==============================] - 3s 56us/sample - loss: 0.2646 - accuracy: 0.9040 - val_loss: 0.3076 - val_accuracy: 0.8926
# Epoch 22/30
# 55000/55000 [==============================] - 3s 54us/sample - loss: 0.2602 - accuracy: 0.9049 - val_loss: 0.3362 - val_accuracy: 0.8724
# Epoch 23/30
# Epoch 24/30
# 55000/55000 [==============================] - 3s 62us/sample - loss: 0.2513 - accuracy: 0.9097 - val_loss: 0.3087 - val_accuracy: 0.8874
# Epoch 25/30
# 55000/55000 [==============================] - 3s 57us/sample - loss: 0.2474 - accuracy: 0.9111 - val_loss: 0.3144 - val_accuracy: 0.8882
# Epoch 26/30
# 55000/55000 [==============================] - 3s 60us/sample - loss: 0.2432 - accuracy: 0.9127 - val_loss: 0.3185 - val_accuracy: 0.8886
# Epoch 27/30
# 55000/55000 [==============================] - 3s 62us/sample - loss: 0.2392 - accuracy: 0.9142 - val_loss: 0.3165 - val_accuracy: 0.8894
# Epoch 28/30
# 55000/55000 [==============================] - 4s 64us/sample - loss: 0.2350 - accuracy: 0.9146 - val_loss: 0.3026 - val_accuracy: 0.8960
# Epoch 29/30
# 55000/55000 [==============================] - 3s 59us/sample - loss: 0.2316 - accuracy: 0.9167 - val_loss: 0.3180 - val_accuracy: 0.8838
# Epoch 30/30
# 55000/55000 [==============================] - 3s 59us/sample - loss: 0.2277 - accuracy: 0.9176 - val_loss: 0.3357 - val_accuracy: 0.8768

import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# y_pred = model.evaluate(x_test, y_test)

x_new = x_test[:3]
y_proba = model.predict(x_new)
print(y_proba.round(2))
# [[0.   0.   0.   0.   0.   0.   0.   0.   0.   0.99]
#  [0.   0.   1.   0.   0.   0.   0.   0.   0.   0.  ]
#  [0.   1.   0.   0.   0.   0.   0.   0.   0.   0.  ]]

y_pred = model.predict_classes(x_new)
print(y_pred)
# [9 2 1]
np.array(class_names)[y_pred]
# array(['Ankle boot', 'Pullover', 'Trouser'], dtype='<U11')
y_new = y_test[:3]
print(y_new)
# [9 2 1]



from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

x_train, x_test, y_train, y_test = train_test_split(housing.data, housing.target, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_val)
x_test = scaler.transform(x_test)



model = Sequential([
    Dense(30, activation="relu", input_shape=x_train.shape[1:]),
    Dense(1)
])
model.compile(loss="mse", optimizer='adam')
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))
# Train on 11610 samples, validate on 3870 samples
# Epoch 1/20
# 11610/11610 [==============================] - 1s 57us/sample - loss: 0.3255 - val_loss: 115735.6243
# Epoch 2/20
# 11610/11610 [==============================] - 0s 36us/sample - loss: 0.3233 - val_loss: 104760.9466
# Epoch 3/20
# 11610/11610 [==============================] - 0s 38us/sample - loss: 0.3210 - val_loss: 108793.9403
# Epoch 4/20
# 11610/11610 [==============================] - 0s 37us/sample - loss: 0.3197 - val_loss: 117315.3350
# Epoch 5/20
# 11610/11610 [==============================] - 0s 36us/sample - loss: 0.3182 - val_loss: 114480.8435
# Epoch 6/20
# 11610/11610 [==============================] - 0s 34us/sample - loss: 0.3171 - val_loss: 99080.4269
# Epoch 7/20
# 11610/11610 [==============================] - 0s 35us/sample - loss: 0.3154 - val_loss: 101056.2749
# Epoch 8/20
# 11610/11610 [==============================] - 0s 36us/sample - loss: 0.3142 - val_loss: 111301.8322
# Epoch 9/20
# 11610/11610 [==============================] - 0s 33us/sample - loss: 0.3145 - val_loss: 106509.8633
# Epoch 10/20
# 11610/11610 [==============================] - 0s 34us/sample - loss: 0.3112 - val_loss: 110706.1073
# Epoch 11/20
# 11610/11610 [==============================] - 0s 34us/sample - loss: 0.3106 - val_loss: 131741.5381
# Epoch 12/20
# 11610/11610 [==============================] - 0s 35us/sample - loss: 0.3101 - val_loss: 99947.0920
# Epoch 13/20
# Epoch 14/20
# 11610/11610 [==============================] - 0s 35us/sample - loss: 0.3078 - val_loss: 113796.6766
# Epoch 15/20
# 11610/11610 [==============================] - 0s 34us/sample - loss: 0.3079 - val_loss: 132141.5407
# Epoch 16/20
# 11610/11610 [==============================] - 0s 35us/sample - loss: 0.3075 - val_loss: 103450.7305
# Epoch 17/20
# 11610/11610 [==============================] - 0s 35us/sample - loss: 0.3072 - val_loss: 134201.3660
# Epoch 18/20
# 11610/11610 [==============================] - 0s 40us/sample - loss: 0.3060 - val_loss: 137520.5067
# Epoch 19/20
# 11610/11610 [==============================] - 0s 38us/sample - loss: 0.3064 - val_loss: 146870.0878
# Epoch 20/20
# 11610/11610 [==============================] - 0s 40us/sample - loss: 0.3046 - val_loss: 160052.7627


# mse_test = model.evaluate(x_test, y_test)

x_new = x_test[:3]
y_pred = model.predict(x_new)
print(y_pred)
# [[0.68981135]
#  [1.7299657 ]
#  [4.2777643 ]]

from keras.layers import concatenate
input_ = Input(shape=x_train.shape[1:])
hidden1 = Dense(30, activation="relu")(input_)
hidden2 = Dense(30, activation="relu")(hidden1)
concat = concatenate([input_, hidden2])
output = Dense(1)(concat)
model = Model(inputs=[input_], outputs=[output])

model.summary()
# Model: "model"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# __________________________________________________________________________________________________
# dense_11 (Dense)                (None, 30)           270         input_1[0][0]
# __________________________________________________________________________________________________
# dense_12 (Dense)                (None, 30)           930         dense_11[0][0]
# __________________________________________________________________________________________________
# concatenate (Concatenate)       (None, 38)           0           input_1[0][0]
#                                                                  dense_12[0][0]
# __________________________________________________________________________________________________
# dense_13 (Dense)                (None, 1)            39          concatenate[0][0]
# ==================================================================================================
# Total params: 1,239
# Trainable params: 1,239
# Non-trainable params: 0
# __________________________________________________________________________________________________

from tensorflow.keras.optimizers import SGD
input_A = Input(shape=[5], name="wide_input")
input_B = Input(shape=[6], name="deep_input")
hidden1 = Dense(30, activation="relu")(input_B)
hidden2 = Dense(30, activation="relu")(hidden1)
concat = concatenate([input_A, hidden2])
output = Dense(1, name="output")(concat)
model = Model(inputs=[input_A, input_B], outputs=[output])

model.compile(loss="mse", optimizer=SGD(lr=1e-3))

x_train_A, x_train_B = x_train[:, :5], x_train[:, 2:]
x_val_A, x_val_B = x_val[:, :5], x_val[:, 2:]
x_test_A, x_test_B = x_test[:, :5], x_test[:, 2:]
x_new_A, x_new_B = x_test_A[:3], x_test_B[:3]

history = model.fit((x_train_A, x_train_B), y_train, epochs=20,
                    validation_data=((x_val_A, x_val_B), y_val))
# Train on 11610 samples, validate on 3870 samples
# Epoch 1/20
# 11610/11610 [==============================] - 1s 63us/sample - loss: 1.6230 - val_loss: 680661.4706
# Epoch 2/20
# 11610/11610 [==============================] - 0s 40us/sample - loss: 0.7938 - val_loss: 633159.1169
# Epoch 3/20
# 11610/11610 [==============================] - 0s 40us/sample - loss: 0.6678 - val_loss: 553271.8911
# Epoch 4/20
# 11610/11610 [==============================] - 0s 38us/sample - loss: 0.6091 - val_loss: 463783.9974
# Epoch 5/20
# 11610/11610 [==============================] - 0s 40us/sample - loss: 0.5682 - val_loss: 389288.6545
# Epoch 6/20
# 11610/11610 [==============================] - 0s 41us/sample - loss: 0.5367 - val_loss: 322059.7117
# Epoch 7/20
# 11610/11610 [==============================] - 1s 46us/sample - loss: 0.5125 - val_loss: 273171.5772
# Epoch 8/20
# 11610/11610 [==============================] - 0s 42us/sample - loss: 0.4937 - val_loss: 226950.9228
# Epoch 9/20
# 11610/11610 [==============================] - 0s 40us/sample - loss: 0.4797 - val_loss: 184557.1815
# Epoch 10/20
# 11610/11610 [==============================] - 0s 42us/sample - loss: 0.4685 - val_loss: 154001.6840
# Epoch 11/20
# 11610/11610 [==============================] - 0s 41us/sample - loss: 0.4600 - val_loss: 128043.9259
# Epoch 12/20
# 11610/11610 [==============================] - 0s 37us/sample - loss: 0.4531 - val_loss: 109650.5776
# Epoch 13/20
# Epoch 14/20
# 11610/11610 [==============================] - 0s 37us/sample - loss: 0.4431 - val_loss: 83858.9447
# Epoch 15/20
# 11610/11610 [==============================] - 0s 41us/sample - loss: 0.4395 - val_loss: 71892.8579
# Epoch 16/20
# 11610/11610 [==============================] - 0s 37us/sample - loss: 0.4367 - val_loss: 62787.7783
# Epoch 17/20
# 11610/11610 [==============================] - 0s 40us/sample - loss: 0.4341 - val_loss: 59622.6654
# Epoch 18/20
# 11610/11610 [==============================] - 0s 37us/sample - loss: 0.4317 - val_loss: 52165.0106
# Epoch 19/20
# 11610/11610 [==============================] - 0s 40us/sample - loss: 0.4296 - val_loss: 52152.4184
# Epoch 20/20
# 11610/11610 [==============================] - 0s 37us/sample - loss: 0.4283 - val_loss: 47827.2988


# mse_test = model.evaluate((x_test_A, x_test_B), y_test)
y_pred = model.predict((x_new_A, x_new_B))
print(y_pred)
# [[0.22860959]
#  [1.9425235 ]
#  [3.370367  ]]

output = Dense(1, name='main_output')(concat)
aux_output = Dense(1, name='aux_output')(hidden2)
model = Model(inputs=[input_A, input_B], outputs = [output, aux_output])

model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer='sgd')

history = model.fit([x_train_A, x_train_B], [y_train, y_train], epochs=20,
                    validation_data=([x_val_A, x_val_B], [y_val, y_val]))
# Train on 11610 samples, validate on 3870 samples
# Epoch 1/20
# 11610/11610 [==============================] - 1s 103us/sample - loss: 0.8433 - main_output_loss: 0.7240 - aux_output_loss: 1.9182 - val_loss: 145073.9041 - val_main_output_loss: 99549.7344 - val_aux_output_loss: 554773.9375
# Epoch 2/20
# 11610/11610 [==============================] - 1s 47us/sample - loss: 0.5445 - main_output_loss: 0.4884 - aux_output_loss: 1.0481 - val_loss: 73906.0710 - val_main_output_loss: 49385.1250 - val_aux_output_loss: 294588.6875
# Epoch 3/20
# 11610/11610 [==============================] - 1s 48us/sample - loss: 0.4744 - main_output_loss: 0.4337 - aux_output_loss: 0.8414 - val_loss: 36741.7635 - val_main_output_loss: 19064.4375 - val_aux_output_loss: 195838.5625
# Epoch 4/20
# 11610/11610 [==============================] - 1s 60us/sample - loss: 0.4885 - main_output_loss: 0.4627 - aux_output_loss: 0.7201 - val_loss: 74384.8509 - val_main_output_loss: 68080.0156 - val_aux_output_loss: 131129.7031
# Epoch 5/20
# 11610/11610 [==============================] - 1s 54us/sample - loss: 0.4309 - main_output_loss: 0.4070 - aux_output_loss: 0.6450 - val_loss: 41169.2350 - val_main_output_loss: 34667.1445 - val_aux_output_loss: 99691.6641
# Epoch 6/20
# 11610/11610 [==============================] - 1s 46us/sample - loss: 0.4212 - main_output_loss: 0.4012 - aux_output_loss: 0.6022 - val_loss: 33266.1481 - val_main_output_loss: 28157.8066 - val_aux_output_loss: 79245.3828
# Epoch 7/20
# 11610/11610 [==============================] - 1s 48us/sample - loss: 0.4132 - main_output_loss: 0.3953 - aux_output_loss: 0.5740 - val_loss: 38891.1327 - val_main_output_loss: 34401.1758 - val_aux_output_loss: 79305.1562
# Epoch 8/20
# 11610/11610 [==============================] - 1s 48us/sample - loss: 0.4097 - main_output_loss: 0.3936 - aux_output_loss: 0.5573 - val_loss: 23101.4924 - val_main_output_loss: 19542.0059 - val_aux_output_loss: 55139.2266
# Epoch 9/20
# 11610/11610 [==============================] - 1s 47us/sample - loss: 0.4029 - main_output_loss: 0.3874 - aux_output_loss: 0.5425 - val_loss: 23151.7998 - val_main_output_loss: 19928.6836 - val_aux_output_loss: 52161.8594
# Epoch 10/20
# 11610/11610 [==============================] - 1s 52us/sample - loss: 0.4002 - main_output_loss: 0.3854 - aux_output_loss: 0.5319 - val_loss: 18661.7554 - val_main_output_loss: 15276.8418 - val_aux_output_loss: 49127.4492
# Epoch 11/20
# 11610/11610 [==============================] - 1s 47us/sample - loss: 0.3939 - main_output_loss: 0.3795 - aux_output_loss: 0.5240 - val_loss: 8305.8283 - val_main_output_loss: 5859.1519 - val_aux_output_loss: 30325.1152
# Epoch 12/20
# 11610/11610 [==============================] - 1s 57us/sample - loss: 0.3899 - main_output_loss: 0.3759 - aux_output_loss: 0.5159 - val_loss: 26033.7690 - val_main_output_loss: 23270.6934 - val_aux_output_loss: 50903.4883
# Epoch 13/20
# Epoch 14/20
# 11610/11610 [==============================] - 1s 53us/sample - loss: 0.3864 - main_output_loss: 0.3733 - aux_output_loss: 0.5031 - val_loss: 24878.9132 - val_main_output_loss: 22394.1094 - val_aux_output_loss: 47243.2578    
# Epoch 15/20
# 11610/11610 [==============================] - 1s 51us/sample - loss: 0.3811 - main_output_loss: 0.3685 - aux_output_loss: 0.4959 - val_loss: 15006.6472 - val_main_output_loss: 12338.9121 - val_aux_output_loss: 39014.5703    
# Epoch 16/20
# 11610/11610 [==============================] - 1s 51us/sample - loss: 0.3842 - main_output_loss: 0.3722 - aux_output_loss: 0.4925 - val_loss: 30418.9724 - val_main_output_loss: 27128.0117 - val_aux_output_loss: 60039.1172    
# Epoch 17/20
# 11610/11610 [==============================] - 1s 50us/sample - loss: 0.3743 - main_output_loss: 0.3623 - aux_output_loss: 0.4851 - val_loss: 25187.1077 - val_main_output_loss: 22193.5957 - val_aux_output_loss: 52128.6133    
# Epoch 18/20
# 11610/11610 [==============================] - 0s 40us/sample - loss: 0.3714 - main_output_loss: 0.3596 - aux_output_loss: 0.4784 - val_loss: 19209.0359 - val_main_output_loss: 15956.0244 - val_aux_output_loss: 48485.0586    
# Epoch 19/20
# 11610/11610 [==============================] - 0s 42us/sample - loss: 0.3691 - main_output_loss: 0.3573 - aux_output_loss: 0.4739 - val_loss: 27819.7622 - val_main_output_loss: 24401.0605 - val_aux_output_loss: 58588.4102    
# Epoch 20/20
# 11610/11610 [==============================] - 1s 44us/sample - loss: 0.3676 - main_output_loss: 0.3562 - aux_output_loss: 0.4701 - val_loss: 39964.4389 - val_main_output_loss: 36568.1758 - val_aux_output_loss: 70533.3516    
# >>>




# total_loss, main_loss, aux_loss = model.evaluate([x_test_A, x_test_B], [y_test, y_test])
y_pred_main, y_pred_aux = model.predict([x_new_A, x_new_B])
print(y_pred_main)
print(y_pred_aux)
# >>> print(y_pred_main)
# [[0.4939264]
#  [1.5884025]
#  [3.4418793]]
# >>> print(y_pred_aux)
# [[0.82863235]
#  [1.7263979 ]
#  [3.1139781 ]]
# >>>

class WideAndDeepModel(Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = Dense(units, activation=activation)
        self.hidden2 = Dense(units, activation=activation)
        self.main_output = Dense(1)
        self.aux_output = Dense(1)
        
    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output

model = WideAndDeepModel()



input_ = Input(shape=x_train.shape[1:])
hidden1 = Dense(30, activation="relu")(input_)
hidden2 = Dense(30, activation="relu")(hidden1)
concat = concatenate([input_, hidden2])
output = Dense(1)(concat)
model = Model(inputs=[input_], outputs=[output])

model.compile(loss="mse", optimizer='adam')

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

check = ModelCheckpoint("my_keras_model.h5", save_best_only=True)
history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_val, y_val),
                    callbacks=[check])

# Train on 11610 samples, validate on 3870 samples
# Epoch 1/10
# 11610/11610 [==============================] - 1s 70us/sample - loss: 1.0773 - val_loss: 101629.5526
# Epoch 2/10
# 11610/11610 [==============================] - 1s 44us/sample - loss: 0.4302 - val_loss: 83498.4715
# Epoch 3/10
# Epoch 4/10
# 11610/11610 [==============================] - 0s 39us/sample - loss: 0.3660 - val_loss: 65108.4801
# Epoch 5/10
# 11610/11610 [==============================] - 0s 41us/sample - loss: 0.3596 - val_loss: 55401.8312
# Epoch 6/10
# 11610/11610 [==============================] - 1s 51us/sample - loss: 0.3566 - val_loss: 94548.8990
# Epoch 7/10
# 11610/11610 [==============================] - 0s 40us/sample - loss: 0.3541 - val_loss: 77586.8465
# Epoch 8/10
# 11610/11610 [==============================] - 1s 48us/sample - loss: 0.3426 - val_loss: 89075.1498
# Epoch 9/10
# 11610/11610 [==============================] - 1s 46us/sample - loss: 0.3429 - val_loss: 90438.7345
# Epoch 10/10
# 11610/11610 [==============================] - 0s 38us/sample - loss: 0.3306 - val_loss: 83336.6543
model = load_model("my_keras_model.h5")
# mse_test = model.evaluate(x_test, y_test)



early = EarlyStopping(patience=10,restore_best_weights=True)
history = model.fit(x_train, y_train, epochs=100,
                    validation_data=(x_val, y_val),
                    callbacks=[check, early])
# mse_test = model.evaluate(X_test, y_test)
# Train on 11610 samples, validate on 3870 samples
# Epoch 1/100
# 11610/11610 [==============================] - 1s 69us/sample - loss: 0.3670 - val_loss: 77200.2580
# Epoch 2/100
# 11610/11610 [==============================] - 0s 42us/sample - loss: 0.3612 - val_loss: 59579.0932
# Epoch 3/100
# 11610/11610 [==============================] - 0s 39us/sample - loss: 0.3613 - val_loss: 75277.2975
# Epoch 4/100
# 11610/11610 [==============================] - 0s 38us/sample - loss: 0.3421 - val_loss: 99561.4219
# Epoch 5/100
# Epoch 6/100
# 11610/11610 [==============================] - 0s 39us/sample - loss: 0.3305 - val_loss: 107561.3865
# Epoch 7/100
# 11610/11610 [==============================] - 0s 40us/sample - loss: 0.3235 - val_loss: 96394.5807
# Epoch 8/100
# 11610/11610 [==============================] - 0s 39us/sample - loss: 0.3272 - val_loss: 68614.1942
# Epoch 9/100
# 11610/11610 [==============================] - 0s 37us/sample - loss: 0.3279 - val_loss: 70333.5200
# Epoch 10/100
# 11610/11610 [==============================] - 0s 39us/sample - loss: 0.3346 - val_loss: 88095.4283
# Epoch 11/100
# 11610/11610 [==============================] - 0s 40us/sample - loss: 0.3184 - val_loss: 77829.3556
# Epoch 12/100
# 11610/11610 [==============================] - 0s 38us/sample - loss: 0.3194 - val_loss: 147604.5472


class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))

val_train_ratio_cb = PrintValTrainRatioCallback()
history = model.fit(x_train, y_train, epochs=1,
                    validation_data=(x_val, y_val),
                    callbacks=[val_train_ratio_cb])



import os
root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
print(run_logdir)
# .\my_logs\run_2020_07_05-17_51_19


model = Sequential()
model.add(Dense(30, activation="relu", input_shape=[8]))
model.add(Dense(30, activation="relu"))
model.add(Dense(1)) 

model.compile(loss="mse", optimizer='sgd')
board = TensorBoard(run_logdir)
history = model.fit(x_train, y_train, epochs=30,
                    validation_data=(x_val, y_val),
                    callbacks=[check, board])

test_logdir = get_run_logdir()
writer = tf.summary.create_file_writer(test_logdir)
with writer.as_default():
    for step in range(1,1000 + 1):
        tf.summary.scalar("my_scalar", np.sin(step / 10), step = step)
        data = (np.random.randn(100) + 2) * step / 100
        tf.summary.histogram("my_hist", data, buckets=50, step = step)
        images = np.random.rand(2,32,32,3)
        tf.summary.image("my_images", images * step / 1000, step=step)
        texts = ["The step is " + str(step), "Its square is " + str(step**2)]
        tf.summary.text("my_text", texts, step=step)
        sine_wave = tf.math.sin(tf.range(12000) / 48000 * 2 * np.pi * step)
        audio = tf.reshape(tf.cast(sine_wave, tf.float32), [1,-1,1])
        tf.summary.audio("my_audio", audio, sample_rate=48000, step=step)





from keras.wrappers.scikit_learn import KerasRegressor
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    input1 = Input(shape=input_shape)
    dense1 = Dense(n_neurons, activation = "relu")(input1)
    for layer in range(n_hidden-1):
        dense1 = Dense(n_neurons, activation="relu")(dense1)
    model.add(Dense(1))
    optimizer = SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

keras_reg = KerasRegressor(build_model)


keras_reg.fit(x_train, y_train, epochs=100,
              validation_data=(x_val, y_val),
              callbacks=[EarlyStopping(patience=10)])
# 363/363 [==============================] - 0s 902us/step - loss: 0.3514 - val_loss: 11522.5615
# Epoch 18/100
# 363/363 [==============================] - 0s 939us/step - loss: 0.3489 - val_loss: 9950.9902
# Epoch 19/100
# 363/363 [==============================] - 0s 1ms/step - loss: 0.3472 - val_loss: 9973.3857
# Epoch 20/100
# 363/363 [==============================] - 0s 979us/step - loss: 0.3467 - val_loss: 9640.0898
# <tensorflow.python.keras.callbacks.History object at 0x0000026283258A08>


from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2),
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param, n_iter=10, cv=3, verbose=2)
rnd_search_cv.fit(x_train, y_train, epochs=100,
                  validation_data=(x_val, y_val),
                  callbacks=[EarlyStopping(patience=10)])

print(rnd_search_cv.best_params_)
print(rnd_search_cv.best_score_)
print(rnd_search_cv.best_estimator_)
print(rnd_search_cv.score(x_test, y_test))

model = rnd_search_cv.best_estimator_.model
print(model)

model.evaluate(x_test, y_test)

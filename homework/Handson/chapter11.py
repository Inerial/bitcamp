from keras.layers import Dense, Input, concatenate, Flatten, LeakyReLU
from keras.initializers import VarianceScaling
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD
from keras.datasets import fashion_mnist

Dense(10, activation="relu", kernel_initializer="he_normal")

init = VarianceScaling(scale=2., mode='fan_avg', distribution='uniform')
Dense(10, activation="relu", kernel_initializer=init)



(x_train, y_train), (x_test,y_test) = fashion_mnist.load_data()

print(x_train.shape)
print(x_train.dtype)
# (60000, 28, 28)
# uint8


x_val, x_train = x_train[:5000] / 255., x_train[5000:] / 255.
y_val, y_train = y_train[:5000], y_train[5000:]
x_test = x_test / 255.



model = Sequential()
model.add(Flatten(input_shape=[28, 28]))
model.add(Dense(300, kernel_initializer="he_normal"))
model.add(LeakyReLU())
model.add(Dense(100, kernel_initializer="he_normal"))
model.add(LeakyReLU())
model.add(Dense(10, activation="softmax"))


model.compile(loss="sparse_categorical_crossentropy", optimizer=SGD(lr=1e-3), metrics=["acc"])

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))


layer = Dense(10,activation = "selu", kernel_initializer="lecun_normal")




from keras.layers import BatchNormalization

model = Sequential()
model.add(Flatten(input_shape=[28, 28]))
model.add(BatchNormalization())
model.add(Dense(300, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(100, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(10, activation="softmax"))

model.summary()
# Model: "sequential_5"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# flatten_5 (Flatten)          (None, 784)               0
# _________________________________________________________________
# batch_normalization (BatchNo (None, 784)               3136
# dense_18 (Dense)             (None, 300)               235500
# _________________________________________________________________
# batch_normalization_1 (Batch (None, 300)               1200
# _________________________________________________________________
# dense_19 (Dense)             (None, 100)               30100
# _________________________________________________________________
# batch_normalization_2 (Batch (None, 100)               400
# _________________________________________________________________
# dense_20 (Dense)             (None, 10)                1010
# =================================================================
# Total params: 271,346
# Trainable params: 268,978
# Non-trainable params: 2,368
# _________________________________________________________________
# >>>


bn1 = model.layers[1]
print([(var.name, var.trainable) for var in bn1.variables])
# [('batch_normalization/gamma:0', True), ('batch_normalization/beta:0', True), ('batch_normalization/moving_mean:0', False), ('batch_normalization/moving_variance:0', False)]
print(bn1.updates)
# [<tf.Operation 'cond/Identity' type=Identity>, <tf.Operation 'cond_1/Identity' type=Identity>]


model.compile(loss="sparse_categorical_crossentropy", optimizer=SGD(lr=1e-3), metrics=["acc"])

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))



optimizer = SGD(clipvalue=1.0)
model.compile(loss='mse', optimizer = optimizer)

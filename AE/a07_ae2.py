from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

from tensorflow.keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]*x_train.shape[2]))/255.
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]*x_test.shape[2]))/255.

model1 = autoencoder(hidden_layer_size=1)
model1.compile(optimizer='adam', loss='mse', metrics=['acc'])
model1.fit(x_train, x_train, epochs=10)
output1 = model1.predict(x_test)

model2 = autoencoder(hidden_layer_size=2)
model2.compile(optimizer='adam', loss='mse', metrics=['acc'])
model2.fit(x_train, x_train, epochs=10)
output2 = model2.predict(x_test)

model3 = autoencoder(hidden_layer_size=4)
model3.compile(optimizer='adam', loss='mse', metrics=['acc'])
model3.fit(x_train, x_train, epochs=10)
output3 = model3.predict(x_test)

model4 = autoencoder(hidden_layer_size=8)
model4.compile(optimizer='adam', loss='mse', metrics=['acc'])
model4.fit(x_train, x_train, epochs=10)
output4 = model4.predict(x_test)

model5 = autoencoder(hidden_layer_size=16)
model5.compile(optimizer='adam', loss='mse', metrics=['acc'])
model5.fit(x_train, x_train, epochs=10)
output5 = model5.predict(x_test)

model6 = autoencoder(hidden_layer_size=32)
model6.compile(optimizer='adam', loss='mse', metrics=['acc'])
model6.fit(x_train, x_train, epochs=10)
output6 = model6.predict(x_test)

# # model.compile(optimizer='adam', loss='mse', metrics=['acc'])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# model.fit(x_train, x_train, epochs=10)
# output = model.predict(x_test)


from matplotlib import pyplot as plt
import random
fig, ax = plt.subplots(7,5, figsize=(15, 15))
datas = [x_test, output1,output2,output3,output4,output5,output6]
labels = ["INPUT", "OUTPUT1", "OUTPUT2", "OUTPUT3", "OUTPUT4", "OUTPUT5", "OUTPUT6"]
random_images = random.sample(range(output1.shape[0]), 5)
for i,axs in enumerate(ax):
    for j , axss in enumerate(axs):
        axss.imshow(datas[i][random_images[j]].reshape(28,28), cmap='gray')
        if i == 0:
            axss.set_ylabel(labels[i], size=40)
        axss.grid(False)
        axss.set_xticks([])
        axss.set_yticks([])

plt.show()
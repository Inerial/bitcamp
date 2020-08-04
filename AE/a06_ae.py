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

model = autoencoder(hidden_layer_size=154)

# model.compile(optimizer='adam', loss='mse', metrics=['acc'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(x_train, x_train, epochs=10)
output = model.predict(x_test)


from matplotlib import pyplot as plt
import random
fig, ((ax1,ax2,ax3,ax4,ax5), (ax6,ax7,ax8,ax9,ax10)) = plt.subplots(2,5, figsize=(20,7))

random_images = random.sample(range(output.shape[0]), 5)
for i , ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
for i , ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(32, (3,3),padding='valid',activation='elu', input_shape=(28,28,1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3,3),padding='valid',activation='elu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128, (3,3),padding='valid',activation='elu'))


    model.add(Conv2DTranspose(128,(3,3),padding='valid',activation='elu'))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2DTranspose(64,(4,4),padding='valid',activation='elu'))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2DTranspose(32,(3,3),padding='valid',activation='elu'))
    model.add(Conv2D(1,(1,1),padding='valid',activation='sigmoid'))
    model.summary()
    return model

from tensorflow.keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1],x_train.shape[2],1))/255.
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1],x_test.shape[2],1))/255.

model = autoencoder(hidden_layer_size=154)

# model.compile(optimizer='adam', loss='mse', metrics=['acc'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(x_train, x_train, epochs=30,batch_size=256, validation_split=0.2)
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

# Epoch 28/30
# 48000/48000 [==============================] - 3s 71us/sample - loss: 0.0639 - acc: 0.8153 - val_loss: 0.0643 - val_acc: 0.8161
# Epoch 29/30
# 48000/48000 [==============================] - 3s 70us/sample - loss: 0.0638 - acc: 0.8153 - val_loss: 0.0644 - val_acc: 0.8161
# Epoch 30/30
# 48000/48000 [==============================] - 3s 70us/sample - loss: 0.0637 - acc: 0.8153 - val_loss: 0.0643 - val_acc: 0.8161
# PS D:\Study> 
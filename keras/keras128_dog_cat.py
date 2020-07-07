from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img, img_to_array

img_dog = load_img('./data/dog_cat/dog.jpg', target_size=(224,224))
img_cat = load_img('./data/dog_cat/cat.jpg', target_size=(224,224))
img_suit = load_img('./data/dog_cat/suit.jpg', target_size=(224,224))
img_yang = load_img('./data/dog_cat/yangpa.jpg', target_size=(224,224))

plt.imshow(img_yang)
plt.imshow(img_cat)

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_suit = img_to_array(img_suit)
arr_yang = img_to_array(img_yang)

print(arr_dog)
print(type(arr_dog))
print(arr_dog.shape)

# RGB ==> BGR
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_suit = preprocess_input(arr_suit)
arr_yang = preprocess_input(arr_yang)
print(arr_dog)

import numpy as np
arr_input = np.stack([arr_dog, arr_cat, arr_suit, arr_yang])

print(arr_input.shape)

model = VGG16()
probs = model.predict(arr_input)

print(probs)
print('probs.shape :', probs.shape)

print(decode_predictions(probs[0:1]))
print(decode_predictions(probs[1:2]))
print(decode_predictions(probs[2:3]))
print(decode_predictions(probs[3:4]))
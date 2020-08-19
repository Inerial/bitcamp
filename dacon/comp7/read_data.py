import numpy as np
import pandas as pd
from skimage.transform import resize
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error as mae
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import matplotlib.pyplot as plt
train = pd.read_csv('./data/dacon/comp7/train.csv', sep=',', header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp7/test.csv', sep=',', header = 0, index_col = 0)
comp = pd.read_csv('./dacon/comp7/comp7_sub.csv', sep=',', header = 0, index_col = 0)

from keras.preprocessing.image import ImageDataGenerator
gen = ImageDataGenerator(
        rotation_range=10, 
        zoom_range = 0.10, 
        shear_range= 0.3,
        # width_shift_range=0.1, 
        # height_shift_range=0.1,
        validation_split = 0.2)  # randomly flip images

x_train = train.values[:, 2:].reshape(-1, 28,28,1)
# x_train[x_train < 0.2] = 2
# x_train[x_train < 0.55] = 0
# x_train[x_train == 2] = 0.2

# for i ,img in enumerate(train.values):
#     if img[0] != 1: continue
#     if img[1] != 'A': continue
#     print(i, img[1])
#     plt.title(str(img[0]) + ' ' + img[1])
#     img = img[2:]
# #     print(img[2:]*100%255)
# #     img[[img < 140]] = 0
#     plt.imshow(img.reshape(28,28).astype(int), cmap='gray')
#     plt.show()
# print(comp.values)
# for i ,img in enumerate(test.values):
# #     if img[0] != 'A': continue
#     print(comp.iloc[i,0])
#     plt.imshow(img[1:].reshape(28,28).astype(int), cmap='gray')
#     plt.title('digit:{} letter:{}'.format(comp.iloc[i,0], img[0]))
#     plt.show()

gen.fit(x_train)
for i, img in enumerate(gen.flow(x_train,y=np.array([train.values[:,0], train.values[:,1]]).T,batch_size=1)):
    if img[1][0][0] != 1: continue
    if img[1][0][1] != 'A': continue
    print(img[1])
    print(img[0][0].shape)
    img[0][0] = img[0][0]*2
    img[0][0][img[0][0] > 255] = 255
    plt.imshow(img[0].reshape(28,28).astype(int) ,cmap='gray')
    plt.show()

# numbers = train.values[:,0] ## 숫자
# letters = train.values[:,1] ## 문자

# letters = np.array([ord(i)-ord('A') for i in letters])
# letters = to_categorical(letters)
# # pd.set_option('display.max_row', 500)
# # print(train.groupby([ train['letter']]).count().iloc[:,0])
# print(letters.shape)

# def generate_data_generator(generator, X1, X2, Y, batch_size, subset):
#     genX1 = generator.flow(X1, X2, seed=7, batch_size=batch_size, subset=subset)
#     genY = generator.flow(X1, Y, seed=7, batch_size=batch_size, subset=subset)
#     while True:
#             Xi1 = genX1.next()
#             Yi = genY.next()
#             yield [Xi1[0][0], Xi1[1][0]], Yi[1][0]

# for [img, letter], y in generate_data_generator(gen, x_train, letters, train.values[:,0], batch_size=1, subset='training'):
#     if letter is not 'A' : continue
#     if y is not  1 : continue
#     print(chr(np.argmax(letter) + ord('A')), y)
#     print(img.shape)
#     plt.imshow(resize(img.reshape(28,28),(56,56)).astype(int) ,cmap='gray')
#     plt.show()

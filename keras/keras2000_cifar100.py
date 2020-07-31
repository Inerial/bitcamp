from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.layers import Conv2D, Dense, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Input, LSTM, UpSampling2D, Conv2DTranspose
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

from keras.applications import ResNet50V2
def extract_resnet(X,y):  
    resnet_model = ResNet50(input_shape=(X.shape[1], X.shape[2], 3), include_top=False) 
    x = Flatten()(resnet_model.layers[-1].output)
    x = Dense(10, activation='softmax')(x)
    
    model = Model(resnet_model.input, x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit(X,y,epochs=10,batch_size=128)
    features_array = resnet_model.predict(X)
    return features_array


(X_train, y_train),(X_test,y_test) = cifar10.load_data()

y_train_0, y_test_0 = (y_train == 0).astype('int')[:,0], (y_test == 0).astype('int')[:,0]
y_train_1, y_test_1 = (y_train == 1).astype('int')[:,0], (y_test == 1).astype('int')[:,0]
y_train_2, y_test_2 = (y_train == 2).astype('int')[:,0], (y_test == 2).astype('int')[:,0]
y_train_3, y_test_3 = (y_train == 3).astype('int')[:,0], (y_test == 3).astype('int')[:,0]
y_train_4, y_test_4 = (y_train == 4).astype('int')[:,0], (y_test == 4).astype('int')[:,0]
y_train_5, y_test_5 = (y_train == 5).astype('int')[:,0], (y_test == 5).astype('int')[:,0]
y_train_6, y_test_6 = (y_train == 6).astype('int')[:,0], (y_test == 6).astype('int')[:,0]
y_train_7, y_test_7 = (y_train == 7).astype('int')[:,0], (y_test == 7).astype('int')[:,0]
y_train_8, y_test_8 = (y_train == 8).astype('int')[:,0], (y_test == 8).astype('int')[:,0]

y_train_res = np_utils.to_categorical(y_train)
y_test_res = np_utils.to_categorical(y_test)

X_train = np.squeeze(extract_resnet(X_train, y_train_res))
X_test = np.squeeze(extract_resnet(X_test, y_test_res))


print(X_train.shape)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm

# Apply standard scaler to output from resnet50
ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_test = ss.transform(X_test)

# Take PCA to reduce feature space dimensionality
pca = PCA(n_components=512, whiten=True)
pca = pca.fit(X_train)
print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train_0, y_val_0 = train_test_split(
    X_train, y_train_0,train_size = 0.8, shuffle=True
)

from sklearn.mixture import GaussianMixture
from sklearn.isotonic import IsotonicRegression
gmm_clf = GaussianMixture(covariance_type='spherical', n_components=18, max_iter=int(1e7),verbose=1)  # Obtained via grid search
gmm_clf.fit(X_train)
log_probs_val = gmm_clf.score_samples(X_val)

isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
isotonic_regressor.fit(log_probs_val, y_val_0)  # y_val is for labels 0 - not food 1 - food (validation set)

# Obtaining results on the test set
log_probs_test = gmm_clf.score_samples(X_test)
test_probabilities = isotonic_regressor.predict(log_probs_test)
test_predictions = [1 if prob >= 0.5 else 0 for prob in test_probabilities]
print(np.square((np.array(test_predictions) - np.array(y_val_0))).sum() )
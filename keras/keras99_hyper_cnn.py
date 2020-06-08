from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Input, Flatten, MaxPool2D
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape) # (60000, 28, 28)
print(y_train.shape) # (60000,)
print(x_test.shape)  # (10000, 28, 28)
print(x_test.shape)  # (10000, 28, 28)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]* x_train.shape[2]* 1) / 255
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]* x_test.shape[2]* 1) / 255

y_train = np_utils.to_categorical(y_train)

print(y_train.shape) # (60000, 10)

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape) # (60000, 28, 28)
print(y_train.shape) # (60000,)
print(x_test.shape)  # (10000, 28, 28)
print(x_test.shape)  # (10000, 28, 28)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1) / 255
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1) / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape) # (60000, 10)

# 2. 모델
## 이렇게 함수형으로 직접 만들어 그리드 서치에 넣으면 파라미터 탐색이 가능하다

def build_model(drop=0.5, optimizer='adam', activations = 'relu', pad = 'same', kernel = (2,2), pool = (2,2)):
    inputs = Input(shape=(x_train.shape[1], x_train.shape[2], 1))
    convs = Conv2D(32, kernel_size=kernel, activation=activations, name='hidden1_cnn', padding=pad)(inputs)
    convs = Dropout(drop)(convs)
    convs = Conv2D(32, kernel_size=kernel, activation=activations, name='hidden2_cnn', padding=pad)(convs)
    convs = Dropout(drop)(convs)
    convs = Conv2D(32, kernel_size=kernel, activation=activations, name='hidden3_cnn', padding=pad)(convs)
    convs = Dropout(drop)(convs)

    convs = MaxPool2D(pool_size=pool)(convs)

    convs = Conv2D(64, kernel_size=kernel, activation=activations, name='hidden1_cnn', padding=pad)(inputs)
    convs = Dropout(drop)(convs)
    convs = Conv2D(64, kernel_size=kernel, activation=activations, name='hidden2_cnn', padding=pad)(convs)
    convs = Dropout(drop)(convs)
    convs = Conv2D(64, kernel_size=kernel, activation=activations, name='hidden3_cnn', padding=pad)(convs)
    convs = Dropout(drop)(convs)

    convs = MaxPool2D(pool_size=pool)(convs)

    convs = Conv2D(128, kernel_size=kernel, activation=activations, name='hidden1_cnn', padding=pad)(inputs)
    convs = Dropout(drop)(convs)
    convs = Conv2D(128, kernel_size=kernel, activation=activations, name='hidden2_cnn', padding=pad)(convs)
    convs = Dropout(drop)(convs)
    convs = Conv2D(128, kernel_size=kernel, activation=activations, name='hidden3_cnn', padding=pad)(convs)
    convs = Dropout(drop)(convs)

    convs = Flatten()(convs)

    denses = Dense(512, activation=activations, name='hidden1')(convs)
    denses = Dropout(drop)(denses)
    denses = Dense(256, activation=activations, name='hidden2')(denses)
    denses = Dropout(drop)(denses)
    denses = Dense(128, activation=activations, name='hidden3')(denses)
    denses = Dropout(drop)(denses)
    output = Dense(y_train.shape[1], activation='softmax', name='output')(denses)
    
    model = Model(inputs=inputs, outputs = output)
    model.compile(loss = 'categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    return model # 콤파일 까지 한 모델 리턴

def create_hyperparameters():
    batches = [100,200,300,400,500]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1,0.5, 5)
    activation = ['relu', 'elu', 'selu']
    pad = ['same', 'valid']
    kernel = [(2,2), (3,3), (4,4)]
    pool = [(2,2), (3,3), (4,4)]
    return {"batch_size" : batches, "optimizer": optimizers, "drop" : dropout , "activations" : activation, "pad" : pad, "kernel" : kernel, "pool" : pool}

## keras형태의 모델 function을 넣어주어 sklearn에 맞춰줌(그냥은 안들어감)
from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_model)

hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(estimator=model, param_distributions=hyperparameters, cv=3, n_iter=10)

search.fit(x_train, y_train, epochs=10)


print("최적의 모델:",search.best_params_)

scores= search.score(x_test, y_test)
print("최종 정답률 : ", scores)


# y_pred= search.predict(x_test)
# y_pred = np.array([i.argmax() for i in y_test])
# print("최종 정답률 : ", accuracy_score(y_test,y_pred))
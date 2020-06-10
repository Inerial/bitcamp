import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.utils import np_utils
from keras import backend

# 1. data
iris = load_iris()
x = iris.data
y = iris.target

y = np_utils.to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=43, shuffle = True, train_size = 0.8
)

def build_model(hidden_layers = 1, nodes = 128, activation = 'relu', drop=0.5, optimizer='adam'):
    backend.clear_session()
    inputs = Input(shape=(x_train.shape[1], ))
    
    denses = Dense(nodes, activation= activation)(inputs)
    denses = Dropout(drop)(denses)
    for i in range(hidden_layers-1):
        denses = Dense(nodes, activation= activation)(denses)
        denses = Dropout(drop)(denses)
    outputs = Dense(y_train.shape[1], activation='softmax')(denses)

    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss = 'categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model # 콤파일 까지 한 모델 리
# grid/random에서의 매개변수
parameters ={'md__hidden_layers' : [1,2,3,5,7,9], 
             'md__nodes' : [64,128,256,512], 
             'md__activation' : ['relu','linear'], 
             'md__drop' : [0.1,0.2,0.3,0.4,0.5], 
             'md__optimizer' : ['adam'],
             "md__batch_size" : [50,10],
             'md__epochs' : [10,50]}

    
models = KerasClassifier(build_fn=build_model)

pipe = Pipeline([
                ('scaler', StandardScaler()), ## 스케일러
                ('md', models)  ## 모델 순
                # 이름   함수 
                # 이름은 파라미터 지정용, 각각 함수에 지정한 파라미터들을 넣어준다
])

model = RandomizedSearchCV(pipe, parameters, cv=5)

model.fit(x_train, y_train)

print("best_model :", model.best_params_)
print("best_model :", model.best_estimator_)
print("acc : ",model.score(x_test,y_test))


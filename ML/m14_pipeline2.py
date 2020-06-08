import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

# 1. data
iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=43, shuffle = True, train_size = 0.8
)

# grid/random에서의 매개변수
parameters = [
    {"svm__C":[1,10,100,1000], "svm__kernel":["linear"]},
    {"svm__C":[1,10,100], "svm__kernel":["rbf"], "svm__gamma": [0.001,0.0001]},
    {"svm__C":[1,100,1000], "svm__kernel":["sigmoid"], "svm__gamma" : [0.001,0.0001]} 
    ## 파라미터 이름 앞에 pipeline에서 지정해준 이름__ 을 붙여주어야 해당 위치 파라미터로 들어간다.
]

# 2. model
# model = SVC()

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

pipe = Pipeline([
                ('scaler', MinMaxScaler()), ## 스케일러
                ('svm', SVC())  ## 모델 순
                # 이름   함수 
                # 이름은 파라미터 지정용, 각각 함수에 지정한 파라미터들을 넣어준다
])
# pipe = make_pipeline(MinMaxScaler(),SVC()) # 함수명 소문자가 이름이 됨

model = RandomizedSearchCV(pipe, parameters, cv=5)

model.fit(x_train, y_train)

print("best_model :", model.best_params_)
print("acc : ",model.score(x_test,y_test))

import sklearn
print("sklearn :",sklearn.__version__)
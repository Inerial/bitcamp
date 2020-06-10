import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# 1. data
iris = load_iris
x = iris.data
y = iris.target

(1,2,3).data

x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=43, shuffle = True, train_size = 0.8
)

# grid/random에서의 매개변수

parameters ={"rf__n_estimators":[1,10,100,1000],
             "rf__max_depth":[None, 1,10,100,1000]}

# parameters ={"randomforestclassifier__n_estimators":[1,10,100,1000],
#              "randomforestclassifier__max_depth":[None, 1,10,100,1000]}

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

pipe = Pipeline([
                ('scaler', MinMaxScaler()), ## 스케일러
                ('rf', RandomForestClassifier(n_jobs=-1,verbose=1))  ## 모델 순
                # 이름   함수 
                # 이름은 파라미터 지정용, 각각 함수에 지정한 파라미터들을 넣어준다
])
# pipe = make_pipeline(MinMaxScaler(),RandomForestClassifier()) # 함수명 소문자가 이름이 됨

model = RandomizedSearchCV(pipe, parameters, cv=5, n_jobs=-1,verbose=1)

model.fit(x_train, y_train)

print("best_model :", model.best_params_)
print("best_model :", model.best_estimator_)
print("acc : ",model.score(x_test,y_test))

import sklearn
print("sklearn :",sklearn.__version__)
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 1. data
iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=43, shuffle = True, train_size = 0.8
)


# 2. model
# model = SVC()

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
pipe = make_pipeline(MinMaxScaler(), SVC())

pipe.fit(x_train, y_train)

print("acc : ", pipe.score(x_test,y_test))

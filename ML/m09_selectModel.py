import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import warnings

warnings.filterwarnings('ignore')

## 1. 데이터
iris = pd.read_csv('./data/csv/iris.csv', sep=',',
                   header = 0, index_col = None)

x = iris.iloc[:,0:4]
y = iris.iloc[:,4]
print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state = 44)

model = all_estimators(type_filter = 'classifier')

for (name, algorithm) in model:
    model = algorithm()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(name, "의 정답률 ", accuracy_score(y_test, y_pred))

import sklearn
print(sklearn.__version__)

## 0.22 버전에서는 제데로 돌아가는 것이 적다.
## 0.20 버전으로 내려주어야한다.
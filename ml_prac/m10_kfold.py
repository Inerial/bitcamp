import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
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

# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state = 44)

kf = KFold(n_splits=5, shuffle=True, random_state = 66, )

allAlgorithms = all_estimators(type_filter = 'classifier')

for (name, algorithm) in allAlgorithms:
    model = algorithm()
    
    scores = cross_val_score(model, x,y, cv = kf)

    print(name, "의 정답률 ", scores)


## kfold cv = 5이면 5조각을 냄, 모든 조각을 각각 검증데이터로 쓰기 위한 데이터군 5개 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import all_estimators
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

## 1. 데이터
boston = pd.read_csv('./data/csv/boston_house_prices.csv', sep=',',
                   header = 1, index_col = None)

x = boston.iloc[:,0:-1]
y = boston.iloc[:,-1]
print(x)
print(y.shape)



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state = 44)

model = all_estimators(type_filter = 'regressor')

for (name, algorithm) in model:
    model = algorithm()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(name, "의 정답률 ", r2_score(y_test, y_pred))



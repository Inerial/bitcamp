import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

## 1. 데이터
wine = pd.read_csv('./ml_prac/csv/winequality-white.csv', sep=';',
                   header = 0, index_col = None)

y = wine['quality']
x = wine.drop('quality', axis=1)

print(x.shape)
print(y.shape)

newlist = []
for i in list(y):
    if i <= 4:
        newlist += [0]
    elif i <= 7:
        newlist += [1]
    else:
        newlist += [2]
## 데이터를 묶어줌
y = newlist

x_train,x_test,y_train,y_test = train_test_split(
    x,y, test_size = 0.2
)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)
acc = model.score(x_test,y_test)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print("acc_score :", accuracy_score(y_test, y_pred))
print("acc : ", acc)

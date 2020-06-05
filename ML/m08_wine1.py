from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # 분류, 회귀
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
import os

## 1. 데이터
path = os.path.dirname(os.path.realpath(__file__))
wine = pd.read_csv(path+'\\csv\\winequality-white.csv', sep=';', header = 0, index_col=None)

x = wine[wine.columns[:-1]]
y = wine['quality']
print(x.shape)
print(y.shape)

x = x.values
y = y.values



x_train, x_test, y_train, y_test = train_test_split(
    x,y, random_state = 66, train_size = 0.8
)

scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# ## 2. 모델
# ModelList = [KNeighborsClassifier(), KNeighborsRegressor(), LinearSVC(), SVC(), RandomForestClassifier(), RandomForestRegressor()]
# Modelnames = ['KNeighborsClassifier', 'KNeighborsRegressor', 'LinearSVC', 'SVC', 'RandomForestClassifier', 'RandomForestRegressor']
# for index, model in enumerate(ModelList):
#     ## 3. 훈련
#     model.fit(x_train, y_train)                              

#     ## 4.평가 예측

#     y_pred = model.predict(x_test)

#     score = model.score(x_test,y_test)

#     print(Modelnames[index],'의 예측 score = ', score)


## randomforest등을 위시한 머신러닝을 쓰는 이유는?
## feature importance때문


# KNeighborsClassifier 의 예측 score =  0.5663265306122449
# KNeighborsRegressor 의 예측 score =  0.3684847346274219
# LinearSVC 의 예측 score =  0.5377551020408163
# SVC 의 예측 score =  0.5663265306122449
# RandomForestClassifier 의 예측 score =  0.7183673469387755
# RandomForestRegressor 의 예측 score =  0.5579007584018427

model = RandomForestClassifier()

model.fit(x_train, y_train)                              

y_pred = model.predict(x_test)

score = model.score(x_test,y_test)

print('RandomForestClassifier의 예측 score = ', score)

# 과적합 방지
# 1. 훈련데이털 량을 늘린다
# 2. 피쳐 수를 줄인다.
# 3. regularization

from xgboost import XGBClassifier, plot_importance
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8,shuffle=True,random_state=66
)

n_estimators = 100
learning_rate = 0.01  ## 학습률
colsample_bytree = 0.85 ## 각 tree마다 샘플링개수
colsample_bylevel = 0.7 ## 0.6~0.9도로 
## 좋은 feature를 내고싶다면 항상 넣도록 하자
max_depth = 5
n_jobs = -1

parameter = [
    {'n_estimators': [100,200,300],
    'learning_rate': [0.1,0.3,0.001,0.01],
    'max_depth': [4,5,6]},
    {'n_estimators': [100,200,300],
    'learning_rate': [0.1,0.3,0.001,0.01],
    'colsample_bytree':[0.6,0.9,1],
    'max_depth': [4,5,6]},
    {'n_estimators': [100,200,300],
    'learning_rate': [0.1,0.3,0.001,0.01],
    'colsample_bylevel': [0.6,0.9,1],
    'max_depth': [4,5,6]}
]


model = GridSearchCV(XGBClassifier(), parameter, cv=5, n_jobs=-1)
model.fit(x_train,y_train)

print("===========================")
print(model.best_estimator_)
print("===========================")
print(model.best_params_)
print("===========================")

score = model.score(x_test,y_test)
print('점수 :', score)



# n_estimators = 100
# learning_rate = 0.01  ## 학습률
# colsample_bytree = 0.85 ## 각 tree마다 샘플링개수
# colsample_bylevel = 0.7
# 점수 : 0.9666666666666667
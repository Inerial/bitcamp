# 과적합 방지
# 1. 훈련데이털 량을 늘린다
# 2. 피쳐 수를 줄인다.
# 3. regularization

from xgboost import XGBRegressor, plot_importance
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8,shuffle=True,random_state=66
)

n_estimators = 10000
learning_rate = 0.0025  ## 학습률
colsample_bytree = 0.68 ## 각 tree마다 샘플링개수
colsample_bylevel = 0.6 ## 0.6~0.9도로 
## 좋은 feature를 내고싶다면 항상 넣도록 하자
max_depth = 5
n_jobs = -1

model = XGBRegressor(max_depth=max_depth, learning_rate= learning_rate,
                      n_estimators= n_estimators, n_jobs=n_jobs,
                      colsample_bylevel = colsample_bylevel,
                      colsample_bytree = colsample_bytree)

model.fit(x_train,y_train)

score = model.score(x_test,y_test)
print('점수 :', score)

print(model.feature_importances_)

plot_importance(model)
# plt.show()


# n_estimators = 10000
# learning_rate = 0.0025  ## 학습률
# colsample_bytree = 0.68 ## 각 tree마다 샘플링개수
# colsample_bylevel = 0.6
# 점수 : 0.9439067672907525
# 과적합 방지
# 1. 훈련데이털 량을 늘린다
# 2. 피쳐 수를 줄인다.
# 3. regularization

from xgboost import XGBClassifier, plot_importance
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,GridSearchCV
import matplotlib.pyplot as plt

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8,shuffle=True,random_state=66
)

n_estimators = 20000
learning_rate = 0.0025  ## 학습률
colsample_bytree = 0.68 ## 각 tree마다 샘플링개수
colsample_bylevel = 0.6 ## 0.6~0.9도로 
## 좋은 feature를 내고싶다면 항상 넣도록 하자
max_depth = 5
n_jobs = -1
parameter = [
    {'n_estimators': [100,150,200,250,300,1000,10000],
    'learning_rate': [0.1,0.001,0.01,0.0025,0.075],
    'max_depth': [4,5,6]},
    {'n_estimators': [100,150,200,250,300,1000,10000],
    'learning_rate': [0.1,0.001,0.01,0.0025,0.075],
    'colsample_bytree':[0.6,0.9,1],
    'max_depth': [4,5,6]},
    {'n_estimators': [100,150,200,250,300,1000,10000],
    'learning_rate': [0.1,0.001,0.01,0.0025,0.075],
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


# XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=0.6, gamma=0, gpu_id=-1,
#               importance_type='gain', interaction_constraints='',
#               learning_rate=0.075, max_delta_step=0, max_depth=5,
#               min_child_weight=1, missing=nan, monotone_constraints='()',
#               n_estimators=1000, n_jobs=0, num_parallel_tree=1,
#               objective='binary:logistic', random_state=0, reg_alpha=0,
#               reg_lambda=1, scale_pos_weight=1, subsample=1,
#               tree_method='exact', validate_parameters=1, verbosity=None)
# ===========================
# {'colsample_bytree': 0.6, 'learning_rate': 0.075, 'max_depth': 5, 'n_estimators': 1000}
# ===========================
# 점수 : 0.9736842105263158
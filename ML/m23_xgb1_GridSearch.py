# 과적합 방지
# 1. 훈련데이털 량을 늘린다
# 2. 피쳐 수를 줄인다.
# 3. regularization

from xgboost import XGBRegressor, plot_importance
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split,GridSearchCV
import matplotlib.pyplot as plt

dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8,shuffle=True,random_state=66
)

parameter = [
    {'n_estimators': [100,150,200,250,300],
    'learning_rate': [0.001,0.01,0.0025,0.075],
    'max_depth': [4,5,6]},
    {'n_estimators': [100,150,200,250,300],
    'learning_rate': [0.001,0.01,0.0025,0.075],
    'colsample_bytree':[0.6,0.68,0.9,1],
    'max_depth': [4,5,6]},
    {'n_estimators': [100,150,200,250,300],
    'learning_rate': [0.001,0.01,0.0025,0.075],
    'colsample_bylevel': [0.6,0.68,0.9,1],
    'max_depth': [4,5,6]}
]

model = GridSearchCV(XGBRegressor(), parameter, cv=5, n_jobs=-1)
model.fit(x_train,y_train)

print("===========================")
print(model.best_estimator_)
print("===========================")
print(model.best_params_)
print("===========================")

score = model.score(x_test,y_test)
print('점수 :', score)

# ===========================
# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.075, max_delta_step=0, max_depth=6,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=300, n_jobs=0, num_parallel_tree=1,
#              objective='reg:squarederror', random_state=0, reg_alpha=0,
#              reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
#              validate_parameters=1, verbosity=None)
# ===========================
# {'colsample_bylevel': 0.6, 'learning_rate': 0.075, 'max_depth': 6, 'n_estimators': 300}
# ===========================
# 점수 : 0.9377827285188635
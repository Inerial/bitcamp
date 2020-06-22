from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# boston = load_boston()
# x = boston.data
# y = boston.target

x, y = load_boston(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8, shuffle = True, random_state=66
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

model = XGBRegressor()
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print("r2 : ", score)


thresholds = np.sort(model.feature_importances_)

print(thresholds)
print(x_train.shape)
print("========================")
import time
start = time.time()
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                               # median 이 둘중 하나 쓰는거 이해하면 사용 가능
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    print(select_x_train.shape)

    # selection_model = XGBRegressor(n_estimators=1000)
    selection_model = RandomizedSearchCV(XGBRegressor(), parameter, cv = 5, n_iter=2)
    selection_model.fit(select_x_train, y_train)

    y_pred = selection_model.predict(select_x_test)
    score = r2_score(y_test,y_pred)
    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))



start2 = time.time()
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                               # median 이 둘중 하나 쓰는거 이해하면 사용 가능
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    print(select_x_train.shape)

    # selection_model = XGBRegressor(n_jobs=7, n_estimators=1000)
    selection_model = RandomizedSearchCV(XGBRegressor(n_jobs=6), parameter, cv = 5, n_iter=2)
    selection_model.fit(select_x_train, y_train)

    y_pred = selection_model.predict(select_x_test)
    score = r2_score(y_test,y_pred)
    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))

print("그냥 걸린 시간 :", start2- start)
print("잡스 걸린 시간 :", time.time()- start2)




## xgboost 에서는 n_jobs가 -1이 적용 안되는것으로 보임
## 최대 코어수 이상을 넣으면 적용이 잘 안된다.
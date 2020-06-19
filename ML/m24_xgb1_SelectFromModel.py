from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

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
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                               # median 이 둘중 하나 쓰는거 이해하면 사용 가능
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    print(select_x_train.shape)

    selection_model = GridSearchCV(XGBRegressor(), parameter, n_jobs=-1, cv = 5)
    selection_model.fit(select_x_train, y_train)

    y_pred = selection_model.predict(select_x_test)
    score = r2_score(y_test,y_pred)
    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))

## 그리스 서치까지 엮고

## 데이콘 적용, 성적 메일로 제출

## 메일 제목 : 이름 등수 ( 김기태 10등 )

## 24의 2,3번 파일 만들기

## 일요일 23시 59분 까지

## xgboost에서 자동 결측치 처리는 자동 보간


# r2 :  0.9221188544655419
# [0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
#  0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
#  0.42848358]
# (404, 13)
# ========================
# (404, 13)
# Thresh=0.001, n=13, R2: 93.78%
# (404, 12)
# Thresh=0.004, n=12, R2: 93.32%
# (404, 11)
# Thresh=0.012, n=11, R2: 92.81%
# (404, 10)
# Thresh=0.012, n=10, R2: 92.80%
# (404, 9)
# Thresh=0.014, n=9, R2: 93.13%
# (404, 8)
# Thresh=0.015, n=8, R2: 93.51%
# (404, 7)
# Thresh=0.018, n=7, R2: 92.74%
# (404, 6)
# Thresh=0.030, n=6, R2: 93.80%
# (404, 5)
# Thresh=0.042, n=5, R2: 92.09%
# (404, 4)
# Thresh=0.052, n=4, R2: 91.18%
# (404, 3)
# Thresh=0.069, n=3, R2: 91.93%
# (404, 2)
# Thresh=0.301, n=2, R2: 81.34%
# (404, 1)
# Thresh=0.428, n=1, R2: 70.59%
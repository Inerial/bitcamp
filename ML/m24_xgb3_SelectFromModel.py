from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# boston = load_boston()
# x = boston.data
# y = boston.target

x, y = load_iris(return_X_y=True)

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

model = XGBClassifier()
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print("acc : ", score)


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

    selection_model = GridSearchCV(XGBClassifier(), parameter, n_jobs=-1, cv = 5)
    selection_model.fit(select_x_train, y_train)

    y_pred = selection_model.predict(select_x_test)
    score = accuracy_score(y_test,y_pred)
    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))

## 그리스 서치까지 엮고

## 데이콘 적용, 성적 메일로 제출

## 메일 제목 : 이름 등수 ( 김기태 10등 )

## 24의 2,3번 파일 만들기

## 일요일 23시 59분 까지

## xgboost에서 자동 결측치 처리는 자동 보간

# acc :  0.9
# [0.01759811 0.02607087 0.33706376 0.6192673 ]
# (120, 4)
# ========================
# (120, 4)
# Thresh=0.018, n=4, R2: 96.67%
# (120, 3)
# Thresh=0.026, n=3, R2: 93.33%
# (120, 2)
# Thresh=0.337, n=2, R2: 96.67%
# (120, 1)
# Thresh=0.619, n=1, R2: 93.33%
# PS D:\Study>
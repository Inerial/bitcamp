from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# boston = load_boston()
# x = boston.data
# y = boston.target

x, y = load_breast_cancer(return_X_y=True)

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


# acc :  0.9736842105263158
# [0.         0.         0.00037145 0.00233393 0.00278498 0.00281184
#  0.00326043 0.00340272 0.00369179 0.00430626 0.0050556  0.00513449
#  0.0054994  0.0058475  0.00639412 0.00769184 0.00775311 0.00903706
#  0.01171023 0.0136856  0.01420499 0.01813928 0.02285903 0.02365488
#  0.03333857 0.06629944 0.09745205 0.11586285 0.22248562 0.28493083]
# (455, 30)
# ========================
# (455, 30)
# Thresh=0.000, n=30, R2: 97.37%
# (455, 30)
# Thresh=0.000, n=30, R2: 97.37%
# (455, 28)
# Thresh=0.000, n=28, R2: 97.37%
# (455, 27)
# Thresh=0.002, n=27, R2: 97.37%
# (455, 26)
# Thresh=0.003, n=26, R2: 97.37%
# (455, 25)
# Thresh=0.003, n=25, R2: 97.37%
# (455, 24)
# Thresh=0.003, n=24, R2: 97.37%
# (455, 23)
# Thresh=0.003, n=23, R2: 97.37%
# (455, 22)
# Thresh=0.004, n=22, R2: 97.37%
# (455, 21)
# Thresh=0.004, n=21, R2: 97.37%
# (455, 20)
# Thresh=0.005, n=20, R2: 96.49%
# (455, 19)
# Thresh=0.005, n=19, R2: 97.37%
# (455, 18)
# Thresh=0.005, n=18, R2: 97.37%
# (455, 17)
# Thresh=0.006, n=17, R2: 96.49%
# (455, 16)
# Thresh=0.006, n=16, R2: 96.49%
# (455, 15)
# Thresh=0.008, n=15, R2: 97.37%
# (455, 14)
# Thresh=0.008, n=14, R2: 96.49%
# (455, 13)
# Thresh=0.009, n=13, R2: 97.37%
# (455, 12)
# Thresh=0.012, n=12, R2: 97.37%
# (455, 11)
# Thresh=0.014, n=11, R2: 97.37%
# (455, 10)
# Thresh=0.014, n=10, R2: 96.49%
# (455, 9)
# Thresh=0.018, n=9, R2: 96.49%
# (455, 8)
# Thresh=0.023, n=8, R2: 96.49%
# (455, 7)
# Thresh=0.024, n=7, R2: 98.25%
# (455, 6)
# Thresh=0.033, n=6, R2: 96.49%
# (455, 5)
# Thresh=0.066, n=5, R2: 96.49%
# (455, 4)
# Thresh=0.097, n=4, R2: 96.49%
# (455, 3)
# Thresh=0.116, n=3, R2: 96.49%
# (455, 2)
# Thresh=0.222, n=2, R2: 90.35%
# (455, 1)
# Thresh=0.285, n=1, R2: 88.60%
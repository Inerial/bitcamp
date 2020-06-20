import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from xgboost import XGBRegressor
test = pd.read_csv('./data/dacon/comp1/test.csv', sep=',', header = 0, index_col = 0)

x_train = np.load('./dacon/comp1/x_train.npy')
y_train = np.load('./dacon/comp1/y_train.npy')
x_pred = np.load('./dacon/comp1/x_pred.npy')

x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, train_size = 0.8, random_state = 66
)
print(x_train.shape)
print(y_train.shape)
# print(x_test.shape)

# 2. model
final_y_pred = []
parameter = [
    {'n_estimators': [100,150,200,250,300,350,400,450,500],
    'learning_rate': [0.001,0.0025,0.05,0.075,0.01],
    'max_depth': [4,5,6]},
    {'n_estimators': [100,150,200,250,300,350,400,450,500],
    'learning_rate': [0.001,0.0025,0.05,0.075,0.01],
    'colsample_bytree':[0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],
    'max_depth': [4,5,6]},
    {'n_estimators': [100,150,200,250,300,350,400,450,500],
    'learning_rate': [0.001,0.0025,0.05,0.075,0.01],
    'colsample_bylevel': [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],
    'max_depth': [4,5,6]}
]
# parameter =[
#     {'n_estimators': [500],
#     'learning_rate': [0.075],
#     'colsample_bylevel': [0.75],
#     'max_depth': [6]},
#     {'n_estimators': [500],
#     'learning_rate': [0.075],
#     'colsample_bytree': [0.75],
#     'max_depth': [6]},
#     {'n_estimators': [500],
#     'learning_rate': [0.075],
#     'max_depth': [6]}
# ]
    
# 모델 컬럼별 4번
for i in range(4):
    model = XGBRegressor()
    model.fit(x_train,y_train[:,i])
    score = model.score(x_test,y_test[:,i])
    print("r2 : ", score)
    thresholds = np.sort(model.feature_importances_)[[i for i in range(0,106,10)]]
    print(thresholds)
    best_score = score
    best_model = model
    best_y_pred = model.predict(x_pred)
    print(best_y_pred.shape)
    for thresh in thresholds:
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                                # median 이 둘중 하나 쓰는거 이해하면 사용 가능
                                                ## 이거 주어준 값 이하의 중요도를 가진 feature를 전부 자르는 파라미터
        select_x_train = selection.transform(x_train)
        select_x_test = selection.transform(x_test)
        select_x_pred = selection.transform(x_pred)

        print(select_x_train.shape)

        selection_model = RandomizedSearchCV(XGBRegressor(), parameter, n_jobs=-1, cv = 5,n_iter=30)
        selection_model.fit(select_x_train, y_train[:,i])

        y_pred = selection_model.predict(select_x_test)
        score = r2_score(y_test[:,i],y_pred)
        print(selection_model.best_params_)
        if score >= best_score:
            best_score = score
            best_model = selection_model
            best_y_pred = selection_model.predict(select_x_pred)
        print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))
    final_y_pred.append(best_y_pred)


final_y_pred = np.array(final_y_pred)
submissions = pd.DataFrame({
    "id": test.index,
    "hhb": final_y_pred[0,:],
    "hbo2": final_y_pred[1,:],
    "ca": final_y_pred[2,:],
    "na": final_y_pred[3,:]
})

submissions.to_csv('./dacon/comp1/comp1_sub.csv', index = False)
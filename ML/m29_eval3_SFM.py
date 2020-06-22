"""
SelectFromModel 에
1. 회귀         m29_eval1
2. 이진 분류    m29_eval2
3. 다중 분류    m29_eval3

1. eval에 'loss'와 다른 지표 1개 더 추가
2. earlyStopping 적용 
3. plot으로 그릴것.

4. 결과는 주석으로 소스 하단에 표시.
"""
from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV

# boston = load_boston()
# x = boston.data
# y = boston.target

x, y = load_iris(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8, shuffle = True, random_state=66
)

parameter = [
    {'n_estimators': [10000],
    'learning_rate': [0.001,0.01,0.0025,0.075],
    'max_depth': [4,5,6]},
    {'n_estimators': [10000],
    'learning_rate': [0.001,0.01,0.0025,0.075],
    'colsample_bytree':[0.6,0.68,0.9,1],
    'max_depth': [4,5,6]},
    {'n_estimators': [10000],
    'learning_rate': [0.001,0.01,0.0025,0.075],
    'colsample_bylevel': [0.6,0.68,0.9,1],
    'max_depth': [4,5,6]}
]
fit_params = {
    'verbose':False,
    'eval_metric': ["mlogloss","merror"],
    'eval_set' :[(x_train, y_train), (x_test,y_test)],
    'early_stopping_rounds': 20
}
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
    fit_params['eval_set'] = [(select_x_train, y_train), (select_x_test,y_test)]
    selection_model.fit(select_x_train, y_train, **fit_params)

    y_pred = selection_model.predict(select_x_test)
    score = accuracy_score(y_test,y_pred)
    print("Thresh=%.3f, n=%d, ACC: %.2f%%" %(thresh, select_x_train.shape[1], score*100.0))


# acc :  0.9
# [0.01759811 0.02607087 0.33706376 0.6192673 ]
# (120, 4)
# ========================
# (120, 4)
# Thresh=0.018, n=4, ACC: 96.67%
# (120, 3)
# Thresh=0.026, n=3, ACC: 96.67%
# (120, 2)
# Thresh=0.337, n=2, ACC: 100.00%
# (120, 1)
# Thresh=0.619, n=1, ACC: 93.33%
# PS D:\Study> 
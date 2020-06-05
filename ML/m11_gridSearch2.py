import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
import warnings
from sklearn.ensemble import RandomForestClassifier
from keras.datasets import mnist
from sklearn.datasets import load_breast_cancer

warnings.filterwarnings('ignore')

## 1. 데이터
cancer = load_breast_cancer()

x = cancer.data
y = cancer.target
print(x)
print(y)

# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state = 44
)

parameters = [
    {"n_estimators": [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90, 100,200,300,400,500,1000]
    , "criterion":["gini", "entropy"],
    "max_depth":[None, 1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90, 100,200,300,400,500,1000]}
]

kfold = KFold(n_splits=5, shuffle=True)
model = GridSearchCV(RandomForestClassifier(verbose=1), parameters, cv=kfold)
model.fit(x_train,y_train)  # train을 5조각 내서 20퍼 검증 , validation_data와 같음

print("최적의 매개변수 :", model.best_estimator_)
print("최적의 매개변수 :", model.best_params_)
y_pred = model.predict(x_test)
print("최종 정답률 : ", accuracy_score(y_test,y_pred))



# 최적의 매개변수 : RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
#             max_depth=10, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=9, n_jobs=None,
#             oob_score=False, random_state=None, verbose=1,
#             warm_start=False)
# 최적의 매개변수 : {'criterion': 'entropy', 'max_depth': 10, 'n_estimators': 9}
# [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
# [Parallel(n_jobs=1)]: Done   9 out of   9 | elapsed:    0.0s finished
# 최종 정답률 :  0.972027972027972
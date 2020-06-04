from sklearn.datasets import load_boston
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # 분류, 회귀
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

## 1. 데이터
data = load_boston()
x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(
    x,y, random_state = 66, train_size = 0.8
)
# scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

## 2. 모델
ModelList = [KNeighborsClassifier(), KNeighborsRegressor(), LinearSVC(), SVC(), RandomForestClassifier(), RandomForestRegressor()]
Modelnames = ['KNeighborsClassifier', 'KNeighborsRegressor', 'LinearSVC', 'SVC', 'RandomForestClassifier', 'RandomForestRegressor']
for index, model in enumerate(ModelList):
    ## 3. 훈련
    try:
        model.fit(x_train, y_train)                              
    except ValueError:
        print("y값이 분류형 데이터가 아님!")
        continue
    ## 4.평가 예측

    y_pred = model.predict(x_test)

    score = model.score(x_test,y_test)

    print(Modelnames[index],'의 예측 score = ', score)
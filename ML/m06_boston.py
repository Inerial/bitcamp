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
# model = KNeighborsClassifier()  ## 분류모델 에러
# model = KNeighborsRegressor() ## 0.7765
# model = LinearSVC() # 에러
# model = SVC() # 에러
# model = RandomForestClassifier() # 에러
model = RandomForestRegressor() # 0.9378

## 3. 훈련
model.fit(x_train, y_train)                        

## 4.평가 예측

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)

print('x_test의 예측결과 :', y_pred)

print("r2 = ", r2)
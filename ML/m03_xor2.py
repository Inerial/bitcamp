from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

## 1. 데이터
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]

## 2. 모델
model = SVC()  ## 선을 두개 그은 형태

## 3. 훈련
model.fit(x_data, y_data)                              

## 4.평가 예측
x_test = [[0,0],[1,0],[0,1],[1,1]]
y_predict = model.predict(x_test)

acc = accuracy_score([0,1,1,0],y_predict) ## = evaluate

print(x_test, '의 예측결과 :', y_predict)
print("acc = ", acc)

## 이런식으로 만들어지는 모델이 많다.
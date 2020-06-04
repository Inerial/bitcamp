from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # 분류, 회귀
from keras.models import Model
from keras.layers import Dense, Input
import numpy as np

## 1. 데이터
x_data = np.array([[0,0],[1,0],[0,1],[1,1]])
y_data = np.array([0,1,1,0])

## 2. 모델
input1 = Input(shape = (2,))
output1 = Dense(20, activation='elu')(input1)
output1 = Dense(20, activation='elu')(output1)
output1 = Dense(20, activation='elu')(output1)
output1 = Dense(20, activation='elu')(output1)
output1 = Dense(1, activation='sigmoid')(output1)
model = Model(inputs = input1, outputs = output1)

## 3. 훈
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['acc'])
model.fit(x_data, y_data, epochs = 200)

## 4.평가 예측
x_test = np.array([[0,0],[1,0],[0,1],[1,1]])

loss, acc = model.evaluate(x_test, np.array([0,1,1,0]))
print('loss :', loss)
print('acc :' , acc)

y_pred = model.predict(x_test)
print(y_pred)

## 히든레이어 붙이면 정말 잘된다
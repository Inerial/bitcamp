import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Lambda
from sklearn.model_selection import train_test_split

x = np.array(range(1,10001))
y = np.array([i%2 for i in x])

x_train, x_test, y_train, y_test = train_test_split(
    x,y,train_size=0.8, shuffle = True
)

model = Sequential()
model.add(Dense(1, activation='elu', input_dim=1))
model.add(Lambda(lambda x: x%2))
model.add(Dense(1, activation='sigmoid'))
## 마지막 activation을 sigmoid같은 0~1사이가 나오는 것을 주어야 한다.
model.compile(loss='binary_crossentropy', optimizer='adamax', metrics = ['acc'])

## binary_crossentropy == 이진분류의 loss값은 이것 하나밖에 없음

model.fit(x_train,y_train, epochs = 1000, batch_size= 128, validation_split=0.2)

loss, acc = model.evaluate(x_test,y_test, batch_size=1)
print('loss :',loss)
print('acc :',acc)

y_pred = model.predict(x)
print('y_pred :', y_pred)
print('y_pred :', np.round(y_pred))


##과제 1
## 최종 출력값 0 or 1 만들기
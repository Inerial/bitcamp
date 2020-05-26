from keras.models import Sequential
from keras.layers import Conv2D ## cnn 레이어함수

## cnn은 자르고 증폭시켜서 사진의 특성을 찾는다
## 
## input_shape 4차원 만약 그림이 만장이면
## x의 shape는 10000, 10, 10, 1 의 4차원 텐서이다.
## input shape 넣을때 행무시 (10, 10, 1)

model = Sequential()
model.add(Conv2D(10, (2,2), padding = 'same', input_shape = (10,10,1)))

## cnn은 데이터를 한번에 분석하지 못한다.

model.add(Conv2D(7, (2,2), padding = 'same'))
model.add(Conv2D(5, (2,2), padding = 'same'))

model.summary()
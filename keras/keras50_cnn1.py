from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,AveragePooling2D## cnn 레이어함수

## cnn은 자르고 증폭시켜서 사진의 특성을 찾는다
## 
## input_shape 4차원 만약 그림이 만장이면
## x의 shape는 10000, 10, 10, 1 의 4차원 텐서이다.
## input shape 넣을때 행무시 (10, 10, 1)

model = Sequential()
model.add(Conv2D( filters = 10, kernel_size = (2,2),
                  input_shape = (10,10,1)))
#(batch_size, height, width, channels)
## cnn은 데이터를 한번에 분석하지 못한다.

model.add(Conv2D(7, (3,3)))
model.add(Conv2D(5, (2,2), padding = 'same'))
model.add(Conv2D(5, (2,2)))
#model.add(Conv2D(5, (2,2), strides=2))
#model.add(Conv2D(5, (2,2), strides=2, padding= 'same'))
model.add(MaxPooling2D(pool_size=4,))
## 출력데이터의 크기를 줄이거나 특정 데이터의 값을 강조하는 용도

model.add(Flatten()) ## 위에서 내려오는 모든 1차원 벡터로 변경시켜줌
#conv2d의 끝은 항상 flatten으로 끝내준다. 항상 == 3차원 shape를 Dense에 넣을수 있는 1차원 dim으로 변환


model.add(Dense(1))
# 이후 데이터로 분류분석 시행

# # padding = 'same'은 kernel_size에 맞춰 묶어주며 줄어든 높이 너비를 output시 높이 넓이를 원래 값과 같게 바꿔준다.

## padding = 'same'은 kernel_size에 맞춰 묶어주며 줄어든 높이 너비를 output시 높이 넓이를 원래 값과 같게 바꿔준다.
## kernel_size (2,2)일 경우 두개씩 묶어 한 픽셀로 만들기 때문에 총 10개에서 한개가 줄어 9개가 된다.
## 만약에 (3,3) 이라면 10개에서 2개가 줄어 8개가 될것이다.

## filter 10 input 1 이면 사진 한장에서 10장이 생성된것과 같다.
# 
# padding = 'same' 의 경우  0의 값을 가지는 값을 추가해 같은 높너비를 유지해준다.
# 가생이는 가운데보다 덜 적합되므로 패딩이 필요하다. (꼭은 아님)
# padding의 디폴트값은 vaild 

## Maxpooling


## parameter == ((kernel_height * kernel_width * input_channel) + 1) * output_channel

model.summary()

## 
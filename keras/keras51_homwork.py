# 2번 첫 답

# x = [1,2,3]
# x = x-1  ## 에러 리스트는 빼기를 못함
# print(x)

import numpy as np
y = np.array([1,2,3,4,5,1,2,3,4,5])
y = y - 1 # 최소값빼주기
print(y)

from keras.utils import np_utils
y = np_utils.to_categorical(y)
print(y)
print(y.shape)


# 2
## 이방식은 데이터의 개수를 세서 갯수만큼 dim을 만들어준다.
## 중간에 빈 공간도 나오지 않음
## 되돌리는것도 가능
## 단 y값을 2차원 행렬로 reshape해주어야 데이터가 들어가진다.
y = np.array([1,2,3,4,6,1,2,3,4,6])
y = y.reshape(10,1)

from sklearn.preprocessing import OneHotEncoder
aaa = OneHotEncoder()
aaa.fit(y)
y = aaa.transform(y).toarray()
aaa.inverse_transform(y) # 되돌리기 가능
np.argmax(y[0])
print(y)
print(y.shape)
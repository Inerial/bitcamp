import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1.데이터
size = 5
a = np.array(range(1,11))
b = np.array([range(1,11), range(2,12)])


def split_x(seq, size):
    if len(seq.shape) == 1:
        aaa = []
        for i in range(len(seq) - size + 1):
            subset = seq[i:i+size]
            aaa.append(subset)
        print(type(aaa))
        aaa = np.array(aaa)
        return aaa.reshape(aaa.shape[0], aaa.shape[1], 1)
    elif len(seq.shape) == 2:
        aaa = []
        for i in range(len(seq.T) - size + 1):
            subset = seq.T[i:i+size]
            aaa.append(subset)
        print(type(aaa))
        return np.array(aaa)
    else :
        print("error")
        return

'''
seq = 나눌 배열, size = time_step
aaa라는 빈 배열을 만듬
seq의길이 - time_step + 1 번째 까지만 반복해야 a 끝숫자까지 딱 맞아떨어짐
i ~ i+time_step 까지의 배열요소를 subset에 저장하여 aaa에 입력
이렇게 묶어준 aaa를 np.array로 묶어서 return


'''


datasetA = split_x(a, size)
datasetB = split_x(b, size)
print("=========================")
print(datasetA)
print(datasetA.shape)
print(datasetB)
print(datasetB.shape)

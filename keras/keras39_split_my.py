import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1.데이터
size = 5
a = np.array(range(1,11))
b = np.array([range(1,11), range(2,12)])


def split_x(seq, size):
    import numpy as np
    if type(seq) != np.ndarray:
        print("입력값이 array가 아님!")
        return
    elif len(seq.shape) == 1:
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
        print("입력값이 3차원 이상!")
        return

'''
아예 3차원 배열로 짤라주는 함수
'''


datasetA = split_x(a, size)
datasetB = split_x(b, size)
print("=========================")
print(datasetA)
print(datasetA.shape)
print(datasetB)
print(datasetB.shape)
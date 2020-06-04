import numpy as np
import time
from numpy.random import rand

N = 150

matA = np.array(rand(N,N))
matB = np.array(rand(N,N))
matC = np.array([[0] * N for _ in range(N)])

start = time.time()

for i in range(N):
    for j in range(N):
        for k in range(N):
            matC[i][j] = matA[i][k] * matB[k][j]

print("파이썬 기능만으로 계산한 결과: %.2f[sec]" %float(time.time() - start))

start = time.time()

matC = np.dot(matA,matB)

print("NumPy를 사용하여 계산한 결과: %.2f[sec]" %float(time.time() - start))


np.array([1,2,3])

np.arange(4)

array_1d = np.array([1,2,3,4,5,6,7,8])
array_2d = np.array([[1,2,3,4],[5,6,7,8]])
array_3d = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])

storages = [24, 3, 4, 23, 10, 12]
print(storages)
print(type(storages))
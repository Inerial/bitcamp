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

# 파이썬 기능만으로 계산한 결과: 3.56[sec]



start = time.time()

matC = np.dot(matA,matB)

print("NumPy를 사용하여 계산한 결과: %.2f[sec]" %float(time.time() - start))
# NumPy를 사용하여 계산한 결과: 0.00[sec]



np.array([1,2,3])

np.arange(4)

array_1d = np.array([1,2,3,4,5,6,7,8])
array_2d = np.array([[1,2,3,4],[5,6,7,8]])
array_3d = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])

storages = [24, 3, 4, 23, 10, 12]
print(storages)
# [24, 3, 4, 23, 10, 12]

print(type(storages))
# [<class 'list']


storages = [1,2,3,4]
new_storages = []
for n in storages:
    n += n
    new_storages.append(n)

print(new_storages)
## [2, 4, 6, 8]



storages = np.array([1,2,3,4])
storages += storages
print(storages)
# [2 4 6 8]



arr = np.array([2, 5, 3, 4, 8])

print(arr+arr) #[ 4 10  6  8 16]
print(arr-arr) #[0 0 0 0 0]
print(arr ** 3) #[  8 125  27  64 512]
print(1/arr) #[0.5        0.2        0.33333333 0.25       0.125     ]



arr = np.arange(10)
print(arr)
# [0 1 2 3 4 5 6 7 8 9]




arr = np.arange(10)
arr[0:3] = 1
print(arr) # [1 1 1 3 4 5 6 7 8 9]




arr = np.arange(10)
print(arr)
# [0 1 2 3 4 5 6 7 8 9]



print(arr[3:6])
# [3 4 5]



arr[3:6] = 24
print(arr)
# [ 0  1  2 24 24 24  6  7  8  9]







arr1 = np.array([1,2,3,4,5])
print(arr1)

arr2 = arr1
arr2[0] = 100

print(arr1)

arr1 = np.array([1,2,3,4,5])
print(arr1)

arr2 = arr1.copy()
arr2[0] = 100

print(arr1)
# [1 2 3 4 5]
# [100   2   3   4   5]
# [1 2 3 4 5]
# [1 2 3 4 5]







arr_List = [x for x in range(10)]
print("리스트형 데이터입니다.")
print("arr_List:", arr_List)
print()

arr_List_copy = arr_List[:]
arr_List_copy[0] = 100

print("리스트 슬라이스는 복사본이 생성되므로 arr_List에는  arr_List_copy의 변경이 반영되지 않습니다.")
print("arr_List:", arr_List)
print()

arr_Numpy = np.arange(10)
print("Numpy의 ndarray 데이터입니다.")
print("arr_Numpy:", arr_Numpy)
print()

arr_Numpy_view = arr_Numpy[:]
arr_Numpy_view = 100

print("Numpy 의 슬라이스는 view(데이터가 저장된 위치의 정보)가 대입되므로  arr_Numpy_view를 변경하면 arr_Numpy에 반영됩니다.")
print("arr_Numpy:",arr_Numpy)
print()

arr_Numpy = np.arange(10)
print("Numpy의 ndarray에서 copy()를 사용한 경우입니다.")
print("arr_Numpy:", arr_Numpy)
print()

arr_Numpy_copy = arr_Numpy[:].copy()
arr_Numpy_copy[0] = 100

print("copy를 사용하면 복사본이 생성되기때문에 arr_Numpy_copy는 arr_Numpy에 영향을 미치지 않습니다.")
print("arr_Numpy:", arr_Numpy)
# 리스트형 데이터입니다.
# arr_List: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 리스트 슬라이스는 복사본이 생성되므로 arr_List에는  arr_List_copy의 변경이 반영되지 않습니다.
# arr_List: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Numpy의 ndarray 데이터입니다.
# arr_Numpy: [0 1 2 3 4 5 6 7 8 9]

# Numpy 의 슬라이스는 view(데이터가 저장된 위치의 정보)가 대입되므로  arr_Numpy_view를 변경하면 arr_Numpy에 반영됩니다.
# arr_Numpy: [0 1 2 3 4 5 6 7 8 9]

# Numpy의 ndarray에서 copy()를 사용한 경우입니다.
# arr_Numpy: [0 1 2 3 4 5 6 7 8 9]

# copy를 사용하면 복사본이 생성되기때문에 arr_Numpy_copy는 arr_Numpy에 영향을 미치지 않습니다.
# arr_Numpy: [0 1 2 3 4 5 6 7 8 9]





arr = np.array([2,4,6,7])
print(arr[np.array([True,True,True,False])])
# [2 4 6]




arr = np.array([2,4,6,7])
print(arr[arr % 3 == 1])
# [4 7]




arr = np.array([2,3,4,5,6,7])
print(arr % 2 == 0)
print(arr[arr % 2 == 0])
# [ True False  True False  True False]
# [2 4 6]





arr = np.array([4, -9,16,-4,20])
print(arr)

arr_abs = np.abs(arr)
print(arr_abs)

print(np.exp(arr_abs))
print(np.sqrt(arr_abs))
# [ 4 -9 16 -4 20]
# [ 4  9 16  4 20]
# [5.45981500e+01 8.10308393e+03 8.88611052e+06 5.45981500e+01
#  4.85165195e+08]
# [2.         3.         4.         2.         4.47213595]





arr1 = [2,5,7,9,5,2]
arr2 = [2,5,8,3,1]

new_arr1 = np.unique(arr1)
print(new_arr1)

print(np.union1d(new_arr1, arr2))
print(np.intersect1d(new_arr1, arr2))
print(np.setdiff1d(new_arr1, arr2))
# [2 5 7 9]
# [1 2 3 5 7 8 9]
# [2 5]
# [7 9]





from numpy.random import randint
arr1 = randint(0, 11, (5,2))
print(arr1)

arr2 = np.random.rand(3)
print(arr2)

# [[ 9  2]
#  [ 8  7]
#  [ 6 10]
#  [ 5  2]
#  [ 4  9]]
# [0.54216101 0.15102646 0.26999373]





arr = np.array([[1,2,3,4],[5,6,7,8]])
print(arr)

print(arr.shape)
print(arr.reshape(4,2))
# [[1 2 3 4]
#  [5 6 7 8]]
# (2, 4)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]




arr = np.array([[1,2,3],[4,5,6]])
print(arr[1])
# [4 5 6]



arr = np.array([[1,2,3],[4,5,6]])
print(arr[1,2])
# 6




arr = np.array([[1,2,3],[4,5,6]])
print(arr[1,1:])
# [5 6]





arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr) 
print(arr[0, 2])
print(arr[1:, :2])

# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]
# 3
# [[4 5]
#  [7 8]]





arr = np.array([[1, 2 ,3], [4, 5, 6]])

print(arr.sum())
print(arr.sum(axis=0))
print(arr.sum(axis=1))

# 21
# [5 7 9]
# [ 6 15]





arr = np.array([[1, 2, 3], [4, 5, 12], [15, 20, 22]])

print(arr.sum(axis=1))
# [ 6 21 57]




arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

print(arr[[3, 2, 0]])
# [[7 8]
#  [5 6]
#  [1 2]]



arr = np.arange(25).reshape(5, 5)
print(arr[[1, 3, 0]])
# [[ 5  6  7  8  9]
#  [15 16 17 18 19]
#  [ 0  1  2  3  4]]



arr = np.arange(10).reshape(2, 5)
print(arr.T) 
# [[0 5]
#  [1 6]
#  [2 7]
#  [3 8]
#  [4 9]]



print(np.transpose(arr))
# [[0 5]
#  [1 6]
#  [2 7]
#  [3 8]
#  [4 9]]




arr = np.array([15, 30, 5])
arr.argsort()
# array([2, 0, 1], dtype=int64)





arr = np.array([[8, 4, 2], [3, 5, 1]])
print(arr.argsort())
print(np.sort(arr))
arr.sort(1)
print(arr)

# [[2 1 0]
#  [2 0 1]]
# [[2 4 8]
#  [1 3 5]]
# [[2 4 8]
#  [1 3 5]]




arr = np.arange(9).reshape(3, 3)

print(np.dot(arr, arr))
vec = arr.reshape(9)

print(np.linalg.norm(vec))

# [[ 15  18  21]
#  [ 42  54  66]
#  [ 69  90 111]]
# 14.2828568570857




arr = np.arange(15).reshape(3, 5)

print(arr.mean(axis=0))
print(arr.sum(axis=1))
print(arr.min())
print(arr.argmax(axis=0))

# [5. 6. 7. 8. 9.]
# [10 35 60]
# 0
# [2 2 2 2 2]




x = np.arange(6).reshape(2, 3)
print(x + 1)
# [[1 2 3]
#  [4 5 6]]



x = np.arange(15).reshape(3, 5)
print(x)

y = np.array([np.arange(5)])
print(y)
z = x - y
print(z)

# [[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
#  [10 11 12 13 14]]
# [[0 1 2 3 4]]
# [[ 0  0  0  0  0]
#  [ 5  5  5  5  5]
#  [10 10 10 10 10]]





np.random.seed(100)

arr = np.random.randint(0, 31, (5, 3))
print(arr)

arr = arr.T
print(arr)

arr1 = arr[:, 1:4]
print(arr1)

arr1.sort(0)
print(arr1)

print(arr1.mean(axis = 0))
# [[ 8 24  3]
#  [ 7 23 15]
#  [16 10 30]
#  [20  2 21]
#  [ 2  2 14]]
# [[ 8  7 16 20  2]
#  [24 23 10  2  2]
#  [ 3 15 30 21 14]]
# [[ 7 16 20]
#  [23 10  2]
#  [15 30 21]]
# [[ 7 10  2]
#  [15 16 20]
#  [23 30 21]]
# [15.         18.66666667 14.33333333]






np.random.seed(0)
def make_image(m, n) :
    image = np.random.randint(0, 6, (m, n)) 
    return image

def change_little(matrix) :
    shape = matrix.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.random.randint(0, 2)==1:
                matrix[i][j] = np.random.randint(0, 6, 1)
    return matrix

image1 = make_image(3, 3)
print(image1)
print()
image2 = change_little(np.copy(image1))
print(image2)
print()
image3 = image2 - image1
print(image3)
print()
image3 = np.abs(image3)
print(image3)

# [[4 5 0]
#  [3 3 3]
#  [1 3 5]]

# [[4 5 0]
#  [3 3 3]
#  [0 5 5]]

# [[ 0  0  0]
#  [ 0  0  0]
#  [-1  2  0]]

# [[0 0 0]
#  [0 0 0]
#  [1 2 0]]
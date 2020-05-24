# 벡터
from typing import List
Vector = List[float]

height_weight_age = [70, 170, 40]

grades = [95, 80, 75, 62]

# 벡터 덧셈
def add(v: Vector, w: Vector):# -> Vector:  ## 리턴값을 vector로 고정시켜주는거 같은데 이상하게 안됨
    """각 성분끼리 더한다"""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i + w_i for v_i, w_i, in zip(v, w)]

assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

# 벡터 뺄셈
def subtract(v: Vector, w: Vector):
    """각 성분끼리 뺀다"""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]

assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]

# 벡터로 구성된 리스트에서 모든 벡터의 각 성분을 더하기
def vector_sum(vectors: List[Vector]):
    """모든 벡터의 각 성분들끼리 더한다"""
    assert vectors, "no vectors provided!"

    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"

    # i번째 결괏값은 모든 벡터의 i번째 성분을 더한 값
    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]

assert vector_sum([1, 2], [3, 4], [5, 6], [7, 8]) == [16, 20]

# 벡터에 스칼라 곱하기
def scalar_multiply(c: float, v: Vector):
    """모든 성분을 c로 곱하기"""
    return [c * v_i for v_i in v]

assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]

# 벡터로 구성된 리스트의 각 성분별 평균 구하기
def vector_mean(vectors: List[Vector]):
    """각 성분별 평균을 계산"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]

# 벡터의 내적
# 내적 = 벡터의 각 성분별 곱한 값을 더해준 것
def dot(v: Vector, w: Vector):
    """v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be same length"
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

assert dot([1, 2, 3], [4, 5, 6]) == 32 

# 내적의 개념을 사용하여 각 성분으 제곱 값의 합
def sum_of_squares(v: Vector):
    """v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

assert sum_of_squares([1, 2, 3]) == 14

# 제곱 값의 합을 이용하여 벡터의 크기 구하기
import math

def magnitude(v: Vector):
    """벡터 v의 크기를 반환"""
    return math.sqrt(sum_of_squares(v))

assert magnitude([3, 4]) == 5

# 두 벡터 간의 거리
def squared_distance(v: Vector, w: Vector):
    """(v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector):
    """벡터 v와 w간의 거리 계산"""
    return math.sqrt(squared_distance(v, w))

# 더욱 깔끔한 코드
def distance(v: Vector, w: Vector):
    return magnitude(subtract(v, w))

             
# Chapter 04 _ 4.2 행렬
# 리스트의 리스트

# 타입 명시를 위한 별칭
Matrix = List[List[float]]

A = [[1, 2, 3],[4, 5, 6]] # 2행 3열

B = [[1, 2],[3, 4],[5, 6]] # 3행 2열

from typing import Tuple

def shape(A: Matrix):
    """(열의 개수, 행의 개수)를 반환"""
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols
assert shape([1, 2, 3], [4, 5, 6]) == (2, 3)

def get_row(A: Matrix, i = int):
    """A의 i번째 행을 반환"""
    return A[i]

def get_column(A: Matrix, j: int):
    """A의 j번째 열을 반환"""
    return [A_i[j] for A_i in A]

# 형태에 맞는 행렬 생성 후 각 원소를 채워주는 함수
from typing import Callable

def make_matrix(num_rows: int, num_cols: int,
                entry_fn: Callable[[int, int], float]):
    """
    (i, j)번째 원소가 entry_fn(i, j)인 num_rows x num_cols 리스트를 반환
    """
    return [[entry_fn(i, j)
            for j in range(num_cols)]
            for i in range(num_rows)]

# 단위행렬 함수
def identify_matrix(n: int):
    """n x n 단위 행렬을 반환"""
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)

assert identify_matrix(5) == [[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]]
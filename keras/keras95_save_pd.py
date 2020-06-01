import pandas as pd
import numpy as np

datasets = pd.read_csv("./data/csv/iris.csv",  ## 주소
                        index_col= None, ## 자동으로 index생성
                        header = 0, ## 첫번째 라인을 헤더로,  None이면 헤더가 없고, 1이면 0은 아예 입력되지 않는다. 여기서 지정한 헤더 아랫부분만 인식
                        sep = ',') ## 구분자 ,

print(datasets)

print(datasets.head()) ## 위에서 5개만 보여주는것
print(datasets.tail()) ## 아래에서 5개만 보여주는것

# print(datasets.values) ## np.array형태로

x = datasets.values[:, :-1]
y = datasets.values[:, -1]

np.save('./data/iris_x.npy', arr = x)
np.save('./data/iris_y.npy', arr = y)
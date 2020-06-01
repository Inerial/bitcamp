import pandas as pd
import numpy as np

datasets = pd.read_csv("./data/csv/iris.csv",  ## 주소
                        index_col= None, ## 자동으로 index생성
                        header = 0, ## 첫번째 라인을 헤더로,  None이면 헤더가 없고, 1이면 0은 아예 입력되지 않는다. 여기서 지정한 헤더 아랫부분만 인식
                        sep = ',') ## 구분자 ,

print(datasets)

print(datasets.head()) ## 위에서 5개만 보여주는것
print(datasets.tail()) ## 아래에서 5개만 보여주는것

# print(datasets.values) ## np.array형태로 안에 가중치가 높은 데이터형태로 전부 바뀌어서 만들어짐
## string형태가 존재시 모든 데이터가 string
## string x , float x, int만 있다면 모든 데이터가 int (boolean도 포함)
## 역시 boolean만 있으면 모든 데이터가 boolean으로 될 것으로 보임

x = datasets.values[:, :-1]
y = datasets.values[:, -1]
print(x) 
print(y)
""" 
np.save('./data/iris_x.npy', arr = x)
np.save('./data/iris_y.npy', arr = y) """
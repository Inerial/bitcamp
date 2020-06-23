import numpy as np
import pandas as pd

def outliers(data, axis= 0):
    quartile_1, quartile_3 = np.percentile(data,[25,75])
    print("1사분위 : ", quartile_1)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)  ## 아래
    upper_bound = quartile_3 + (iqr * 1.5)  ## 위
    return np.where((data > upper_bound) | (data < lower_bound))
   
## 범위를 벗어나는 데이터 위치 리턴

a1 = np.array([1,2,3,4,10000,6,7,5000,90,100])

f = outliers(a1)
print("이상치의 위치 : \n", f)


## 데이터에 nan값이 들어가면 해당 라인 죄다 처리
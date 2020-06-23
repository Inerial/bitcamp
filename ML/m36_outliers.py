import numpy as np
import pandas as pd

def outliers(data, axis= 0):
    if type(data) == pd.DataFrame:
        data = data.values
    if len(data.shape) == 1:
        quartile_1, quartile_3 = np.percentile(data,[25,75])
        print("1사분위 : ", quartile_1)
        print("3사분위 : ", quartile_3)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)  ## 아래
        upper_bound = quartile_3 + (iqr * 1.5)  ## 위
        return np.where((data > upper_bound) | (data < lower_bound))
    else:
        output = []
        for i in range(data.shape[axis]):
            if axis == 0:
                quartile_1, quartile_3 = np.percentile(data[i, :],[25,75])
            else:
                quartile_1, quartile_3 = np.percentile(data[:, i],[25,75])
            print("1사분위 : ", quartile_1)
            print("3사분위 : ", quartile_3)
            iqr = quartile_3 - quartile_1
            lower_bound = quartile_1 - (iqr * 1.5)  ## 아래
            upper_bound = quartile_3 + (iqr * 1.5)  ## 위
            if axis == 0:
                output.append(np.where((data[i, :] > upper_bound) | (data[i, :] < lower_bound))[0])
            else:
                output.append(np.where((data[:, i] > upper_bound) | (data[:, i] < lower_bound))[0])
    return np.array(output)
## 범위를 벗어나는 데이터 위치 리턴

a1 = np.array([[1,2,3,4,10000,6,7,5000,90,100],[10000,1,2,3,1000,500,2,3,4,np.nan]])
a = pd.DataFrame([[1,2,3,4,10000,6,7,5000,90,100],[10000,1,2,3,1000,500,2,3,4,7]])

b = outliers(a.iloc[0])
c = outliers(a)
d = outliers(a1[0])
f = outliers(a1)
print("이상치의 위치 : \n", b)
print("이상치의 위치 : \n", c)
print("이상치의 위치 : \n", d)
print("이상치의 위치 : \n", f)


## 데이터에 nan값이 들어가면 해당 라인 죄다 처리
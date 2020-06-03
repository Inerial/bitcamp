import numpy as np
import pandas as pd

samsung = pd.read_csv('./test_samsung/csv/samsung.csv',
                      header = 0,
                      index_col=0,
                      sep=',',
                      encoding='CP949')
                      
hite = pd.read_csv('./test_samsung/csv/hite.csv',
                      header = 0,
                      index_col=0,
                      sep=',',
                      encoding='CP949')

print(samsung)
print(hite)
print(samsung.shape)
print(hite.shape)
'''

## Nan 제거1
samsung = samsung.dropna(axis=0)
#print(samsung)
# print(samsung.shape)
hite = hite.fillna(method = "bfill")
hite = hite.dropna(axis=0)
# print(hite)
 '''''' 
## Nan 제거2
hite = hite[0:509]
# hite.iloc[0,1:5] = [10,20,30,40]
hite.loc['2020-06-02', '고가' : '거래량'] = ['100','200','300','400'] ## 거래량도 들어간다.
 '''
## Nan 제거 내꺼
samsung = samsung[samsung.index.notna()]
hite = hite[hite.index.notna()]
hite = hite.fillna(method = "bfill") 


## 결측치 제거에 회귀형으로 채워넣곤 한다. bfill도 나쁜건 아님
## 결측치 제거에 이것저것 함정을 파기 쉬움, 데이터를 잘 볼것
# print(hite)

## 삼성과 하이트의 정렬을 오름차순으로 변경
samsung = samsung.sort_values(['일자'], ascending = True)
hite = hite.sort_values(['일자'], ascending = True)
# print(samsung)
# print(hite)

# 콤마제거, 문자를 정수로 형변환
for i in range(len(samsung.index)):
    samsung.iloc[i,0] = int(samsung.iloc[i,0].replace(',',''))
print(samsung)

for i in range(len(hite.index)):
    for j in range(len(hite.iloc[i])):
        hite.iloc[i,j] = int(hite.iloc[i,j].replace(',',''))

print(hite)
print(type(hite.iloc[1,1]))

print(type(samsung.iloc[1]))
print(type(samsung.iloc[1,0]))

print(samsung.shape)
print(hite.shape)

samsung = samsung.values
hite = hite.values

np.save('./data/samsung.npy', arr = samsung)
np.save('./data/hite.npy', arr=hite)
import pandas as pd

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)

df.columns=["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium","Total phenols", "Flavanoids", "Nonflavanoid phenols",
"Proanthocyanins","Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]

print(df)

#         Alcohol  Malic acid   Ash  Alcalinity of ash  Magnesium  Total phenols  Flavanoids  Nonflavanoid phenols  Proanthocyanins  Color intensity   Hue  OD280/OD315 of diluted wines  Proline
# 0    1    14.23        1.71  2.43               15.6        127           2.80        3.06                  0.28             2.29             5.64  1.04                          3.92     1065
# 1    1    13.20        1.78  2.14               11.2        100           2.65        2.76                  0.26             1.28             4.38  1.05                          3.40     1050
# 2    1    13.16        2.36  2.67               18.6        101           2.80        3.24                  0.30             2.81             5.68  1.03                          3.17     1185
# 3    1    14.37        1.95  2.50               16.8        113           3.85        3.49                  0.24             2.18             7.80  0.86                          3.45     1480
# 4    1    13.24        2.59  2.87               21.0        118           2.80        2.69                  0.39             1.82             4.32  1.04                          2.93      735
# ..  ..      ...         ...   ...                ...        ...            ...         ...                   ...              ...              ...   ...                           ...      ...
# 173  3    13.71        5.65  2.45               20.5         95           1.68        0.61                  0.52             1.06             7.70  0.64                          1.74      740
# 174  3    13.40        3.91  2.48               23.0        102           1.80        0.75                  0.43             1.41             7.30  0.70                          1.56      750
# 175  3    13.27        4.28  2.26               20.0        120           1.59        0.69                  0.43             1.35            10.20  0.59                          1.56      835
# 176  3    13.17        2.59  2.37               20.0        120           1.65        0.68                  0.53             1.46             9.30  0.60                          1.62      840
# 177  3    14.13        4.10  2.74               24.5         96           2.05        0.76                  0.56             1.35             9.20  0.61                          1.60      560

# [178 rows x 14 columns]

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header = None)
df.columns = ["sepal length", "sepal width", "petal length", "petal width", "class"]

print(df)

#      sepal length  sepal width  petal length  petal width           class
# 0             5.1          3.5           1.4          0.2     Iris-setosa
# 1             4.9          3.0           1.4          0.2     Iris-setosa
# 2             4.7          3.2           1.3          0.2     Iris-setosa
# 3             4.6          3.1           1.5          0.2     Iris-setosa
# 4             5.0          3.6           1.4          0.2     Iris-setosa
# ..            ...          ...           ...          ...             ...
# 145           6.7          3.0           5.2          2.3  Iris-virginica
# 146           6.3          2.5           5.0          1.9  Iris-virginica
# 147           6.5          3.0           5.2          2.0  Iris-virginica
# 148           6.2          3.4           5.4          2.3  Iris-virginica
# 149           5.9          3.0           5.1          1.8  Iris-virginica

# [150 rows x 5 columns]



import csv

# with 문을 사용해 파일을 처리
with open("csv0.csv", "w") as csvfile:
    writer = csv.writer(csvfile, lineterminator="\n")

    writer.writerow(["city", "year", "season"])
    writer.writerow(["Nagano", 1998, "winter"])
    writer.writerow(["Sydney", 2000, "summer"])
    writer.writerow(["Salt Lake City", 2002, "winter"])
    writer.writerow(["Athens", 2004, "summer"])
    writer.writerow(["Torino", 2006, "winter"])
    writer.writerow(["Beijing", 2008, "summer"])
    writer.writerow(["Vancouver", 2010, "winter"])
    writer.writerow(["London", 2012, "summer"])
    writer.writerow(["Sochi", 2014, "winter"])
    writer.writerow(["Rio de Janeiro", 2016, "summer"])


with open("csv1.csv", "w") as csvfile:
    # writer() 메서드의 인수로 csvfile과 개행(줄바꿈) 코드(\n)를 지정합니다
    writer = csv.writer(csvfile, lineterminator="\n")

    # writerow(리스트) 로 행을 추가합니다
    writer.writerow(["a", "b", "c", "d"])
    writer.writerow(["1", "2", "3", "4"])
    writer.writerow(["1", "2", "3", "4"])
    writer.writerow(["1", "2", "3", "4"])
    writer.writerow(["1", "2", "3", "4"])




data = {"city": ["Nagano", "Sydney", "Salt Lake City", "Athens", "Torino", "Beijing", "Vancouver", "London", "Sochi", "Rio de Janeiro"],
        "year": [1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016],
        "season": ["winter", "summer", "winter", "summer", "winter", "summer", "winter", "summer", "winter", "summer"]}

df = pd.DataFrame(data)
df.to_csv("csv2.csv")





data = {"OS": ["Machintosh", "Windows", "Linux"],
        "release": [1984, 1985, 1991],
        "country": ["US", "US", ""]}

df = pd.DataFrame(data)
df.to_csv("OSlist.csv")




from pandas import Series, DataFrame
attri_data1 = {"ID": ["100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
               "city": ["서울", "부산", "대전", "광주", "서울", "서울", "부산", "대전", "광주", "서울"],
               "birth_year": [1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name": ["영이", "순돌", "짱구", "태양", "션", "유리", "현아", "태식", "민수", "호식"]}

attri_data_frame1 = DataFrame(attri_data1)
attri_data2 = {"ID": ["107", "109"],
               "city": ["봉화", "전주"],
               "birth_year": [1994, 1988]}

attri_data_frame2 = DataFrame(attri_data2)

attri_data_frame1 = attri_data_frame1.append(attri_data_frame2).sort_values(by="ID", ascending=True).reset_index(drop=True)

print(attri_data_frame1)

#      ID city  birth_year name
# 0   100   서울        1990   영이
# 1   101   부산        1989   순돌
# 2   102   대전        1992   짱구
# 3   103   광주        1997   태양
# 4   104   서울        1982    션
# 5   106   서울        1991   유리
# 6   107   봉화        1994  NaN
# 7   108   부산        1988   현아
# 8   109   전주        1988  NaN
# 9   110   대전        1990   태식
# 10  111   광주        1995   민수
# 11  113   서울        1981   호식




import numpy as np
from numpy import nan as NA

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1,0] = NA
sample_data_frame.iloc[2,2] = NA
sample_data_frame.iloc[5:,3] = NA 

print(sample_data_frame)

#           0         1         2         3
# 0  0.264130  0.218719  0.340276  0.036675
# 1       NaN  0.145854  0.101939  0.513876
# 2  0.598400  0.890832       NaN  0.083968
# 3  0.127188  0.389194  0.508909  0.983787
# 4  0.929232  0.138803  0.384833  0.422301
# 5  0.366386  0.966785  0.244057       NaN
# 6  0.880003  0.277728  0.593255       NaN
# 7  0.956023  0.941701  0.898553       NaN
# 8  0.223694  0.695133  0.400819       NaN
# 9  0.436846  0.485972  0.887083       NaN



print(sample_data_frame.dropna())
#           0         1         2         3
# 0  0.187740  0.299597  0.229128  0.130411
# 3  0.881991  0.544727  0.323403  0.586036
# 4  0.474114  0.850068  0.278001  0.540407


print(sample_data_frame[[0,1,2]].dropna())

#           0         1         2
# 0  0.187740  0.299597  0.229128
# 3  0.881991  0.544727  0.323403
# 4  0.474114  0.850068  0.278001
# 5  0.392999  0.610070  0.958267
# 6  0.827579  0.810924  0.581162
# 7  0.930635  0.528506  0.546035
# 8  0.458837  0.942695  0.119758
# 9  0.701224  0.638370  0.168312




np.random.seed(0)
sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1, 0] = NA
sample_data_frame.iloc[2, 2] = NA
sample_data_frame.iloc[5:, 3] = NA

print(sample_data_frame[[0, 2]].dropna())

#           0         2
# 0  0.548814  0.602763
# 3  0.568045  0.071036
# 4  0.020218  0.778157
# 5  0.978618  0.461479
# 6  0.118274  0.143353
# 7  0.521848  0.264556
# 8  0.456150  0.018790
# 9  0.612096  0.943748


sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1,0] = NA
sample_data_frame.iloc[2,2] = NA
sample_data_frame.iloc[5:,3] = NA


print(sample_data_frame.fillna(0))

#           0         1         2         3
# 0  0.359508  0.437032  0.697631  0.060225
# 1  0.000000  0.670638  0.210383  0.128926
# 2  0.315428  0.363711  0.000000  0.438602
# 3  0.988374  0.102045  0.208877  0.161310
# 4  0.653108  0.253292  0.466311  0.244426
# 5  0.158970  0.110375  0.656330  0.000000
# 6  0.196582  0.368725  0.820993  0.000000
# 7  0.837945  0.096098  0.976459  0.000000
# 8  0.976761  0.604846  0.739264  0.000000
# 9  0.282807  0.120197  0.296140  0.000000



print(sample_data_frame.fillna(method="ffill"))
#           0         1         2         3
# 0  0.359508  0.437032  0.697631  0.060225
# 1  0.359508  0.670638  0.210383  0.128926
# 2  0.315428  0.363711  0.210383  0.438602
# 3  0.988374  0.102045  0.208877  0.161310
# 4  0.653108  0.253292  0.466311  0.244426
# 5  0.158970  0.110375  0.656330  0.244426
# 6  0.196582  0.368725  0.820993  0.244426
# 7  0.837945  0.096098  0.976459  0.244426
# 8  0.976761  0.604846  0.739264  0.244426
# 9  0.282807  0.120197  0.296140  0.244426



np.random.seed(0)

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1, 0] = NA
sample_data_frame.iloc[6:, 2] = NA

print(sample_data_frame.fillna(method="ffill"))
#           0         1         2         3
# 0  0.548814  0.715189  0.602763  0.544883
# 1  0.548814  0.645894  0.437587  0.891773
# 2  0.963663  0.383442  0.791725  0.528895
# 3  0.568045  0.925597  0.071036  0.087129
# 4  0.020218  0.832620  0.778157  0.870012
# 5  0.978618  0.799159  0.461479  0.780529
# 6  0.118274  0.639921  0.461479  0.944669
# 7  0.521848  0.414662  0.461479  0.774234
# 8  0.456150  0.568434  0.461479  0.617635
# 9  0.612096  0.616934  0.461479  0.681820



sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1, 0] = NA
sample_data_frame.iloc[2, 2] = NA
sample_data_frame.iloc[5:, 3] = NA

print(sample_data_frame.fillna(sample_data_frame.mean()))
#           0         1         2         3
# 0  0.359508  0.437032  0.697631  0.060225
# 1  0.529943  0.670638  0.210383  0.128926
# 2  0.315428  0.363711  0.563599  0.438602
# 3  0.988374  0.102045  0.208877  0.161310
# 4  0.653108  0.253292  0.466311  0.244426
# 5  0.158970  0.110375  0.656330  0.206698
# 6  0.196582  0.368725  0.820993  0.206698
# 7  0.837945  0.096098  0.976459  0.206698
# 8  0.976761  0.604846  0.739264  0.206698
# 9  0.282807  0.120197  0.296140  0.206698


np.random.seed(0)

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

sample_data_frame.iloc[1, 0] = NA
sample_data_frame.iloc[6:, 2] = NA

print(sample_data_frame.fillna(sample_data_frame.mean()))

#           0         1         2         3
# 0  0.548814  0.715189  0.602763  0.544883
# 1  0.531970  0.645894  0.437587  0.891773
# 2  0.963663  0.383442  0.791725  0.528895
# 3  0.568045  0.925597  0.071036  0.087129
# 4  0.020218  0.832620  0.778157  0.870012
# 5  0.978618  0.799159  0.461479  0.780529
# 6  0.118274  0.639921  0.523791  0.944669
# 7  0.521848  0.414662  0.523791  0.774234
# 8  0.456150  0.568434  0.523791  0.617635
# 9  0.612096  0.616934  0.523791  0.681820



df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)
df.columns=["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium","Total phenols", "Flavanoids", "Nonflavanoid phenols", 
            "Proanthocyanins","Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
print(df["Alcohol"].mean())
# 13.000617977528083



df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)
df.columns=["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium","Total phenols", "Flavanoids", "Nonflavanoid phenols", 
            "Proanthocyanins","Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]

print(df["Magnesium"].mean())
# 99.74157303370787




dupli_data = DataFrame({"col1":[1, 1, 2, 3, 4, 4, 6, 6], 
                        "col2":["a", "b", "b", "b", "c", "c", "b", "b"]}) 

print(dupli_data)

#    col1 col2
# 0     1    a
# 1     1    b
# 2     2    b
# 3     3    b
# 4     4    c
# 5     4    c
# 6     6    b
# 7     6    b

print(dupli_data.duplicated())

# 0    False
# 1    False
# 2    False
# 3    False
# 4    False
# 5     True
# 6    False
# 7     True
# dtype: bool


print(dupli_data.drop_duplicates())

#    col1 col2
# 0     1    a
# 1     1    b
# 2     2    b
# 3     3    b
# 4     4    c
# 6     6    b

dupli_data = DataFrame({"col1":[1, 1, 2, 3, 4, 4, 6, 6, 7, 7, 7, 8, 9, 9],
                        "col2":["a", "b", "b", "b", "c", "c", "b", "b", "d", "d", "c", "b", "c", "c"]})

print(dupli_data.drop_duplicates())
#     col1 col2
# 0      1    a
# 1      1    b
# 2      2    b
# 3      3    b
# 4      4    c
# 6      6    b
# 8      7    d
# 10     7    c
# 11     8    b
# 12     9    c

attri_data1 = {"ID": ["100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
               "city": ["서울", "부산", "대전", "광주", "서울", "서울", "부산", "대전", "광주", "서울"],
               "birth_year": [1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name": ["영이", "순돌", "짱구", "태양", "션", "유리", "현아", "태식", "민수", "호식"]}
attri_data_frame1 = DataFrame(attri_data1)

print(attri_data_frame1)

#     ID city  birth_year name
# 0  100   서울        1990   영이
# 1  101   부산        1989   순돌
# 2  102   대전        1992   짱구
# 3  103   광주        1997   태양
# 4  104   서울        1982    션
# 5  106   서울        1991   유리
# 6  108   부산        1988   현아
# 7  110   대전        1990   태식
# 8  111   광주        1995   민수
# 9  113   서울        1981   호식




city_map ={"서울":"서울", 
           "광주":"전라도", 
           "부산":"경상도", 
           "대전":"충청도"}
print(city_map)
# {'서울': '서울', '광주': '전라도', '부산': '경상도', '대전': '충청도'}\




attri_data_frame1["region"] = attri_data_frame1["city"].map(city_map)
print(attri_data_frame1)

#     ID city  birth_year name region
# 0  100   서울        1990   영이     서울
# 1  101   부산        1989   순돌    경상도
# 2  102   대전        1992   짱구    충청도
# 3  103   광주        1997   태양    전라도
# 4  104   서울        1982    션     서울
# 5  106   서울        1991   유리     서울
# 6  108   부산        1988   현아    경상도
# 7  110   대전        1990   태식    충청도
# 8  111   광주        1995   민수    전라도
# 9  113   서울        1981   호식     서울

attri_data1 = {"ID": ["100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
               "city": ["서울", "부산", "대전", "광주", "서울", "서울", "부산", "대전", "광주", "서울"],
               "birth_year": [1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name": ["영이", "순돌", "짱구", "태양", "션", "유리", "현아", "태식", "민수", "호식"]}

attri_data_frame1 = DataFrame(attri_data1)

MS_map = {"서울":"중부",
          "광주":"남부",
          "부산":"남부",
          "대전":"중부"}

attri_data_frame1["MS"] = attri_data_frame1["city"].map(MS_map)

print(attri_data_frame1)
#     ID city  birth_year name  MS
# 0  100   서울        1990   영이  중부
# 1  101   부산        1989   순돌  남부
# 2  102   대전        1992   짱구  중부
# 3  103   광주        1997   태양  남부
# 4  104   서울        1982    션  중부
# 5  106   서울        1991   유리  중부
# 6  108   부산        1988   현아  남부
# 7  110   대전        1990   태식  중부
# 8  111   광주        1995   민수  남부
# 9  113   서울        1981   호식  중부

attri_data1 = {"ID": ["100", "101", "102", "103", "104", "106", "108", "110", "111", "113"],
               "city": ["서울", "부산", "대전", "광주", "서울", "서울", "부산", "대전", "광주", "서울"],
               "birth_year": [1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name": ["영이", "순돌", "짱구", "태양", "션", "유리", "현아", "태식", "민수", "호식"]}

attri_data_frame1 = DataFrame(attri_data1)


birth_year_bins = [1980, 1985, 1990, 1995, 2000]

birth_year_cut_data = pd.cut(attri_data_frame1.birth_year, birth_year_bins)
print(birth_year_cut_data)


# 0    (1985, 1990]
# 1    (1985, 1990]
# 2    (1990, 1995]
# 3    (1995, 2000]
# 4    (1980, 1985]
# 5    (1990, 1995]
# 6    (1985, 1990]
# 7    (1985, 1990]
# 8    (1990, 1995]
# 9    (1980, 1985]
# Name: birth_year, dtype: category
# Categories (4, interval[int64]): [(1980, 1985] < (1985, 1990] < (1990, 1995] < (1995, 2000]]

print(pd.value_counts(birth_year_cut_data))

# (1985, 1990]    4
# (1990, 1995]    3
# (1980, 1985]    2
# (1995, 2000]    1
# Name: birth_year, dtype: int64

group_names = ["first1980", "second1980", "first1990", "second1990"]
birth_year_cut_data = pd.cut(attri_data_frame1.birth_year,birth_year_bins,labels = group_names)
print(pd.value_counts(birth_year_cut_data))

# second1980    4
# first1990     3
# first1980     2
# second1990    1
# Name: birth_year, dtype: int64

print(pd.cut(attri_data_frame1.birth_year, 2))

# 0      (1989.0, 1997.0]
# 1    (1980.984, 1989.0]
# 2      (1989.0, 1997.0]
# 3      (1989.0, 1997.0]
# 4    (1980.984, 1989.0]
# 5      (1989.0, 1997.0]
# 6    (1980.984, 1989.0]
# 7      (1989.0, 1997.0]
# 8      (1989.0, 1997.0]
# 9    (1980.984, 1989.0]
# Name: birth_year, dtype: category
# Categories (2, interval[float64]): [(1980.984, 1989.0] < (1989.0, 1997.0]]

attri_data1 = {"ID": [100, 101, 102, 103, 104, 106, 108, 110, 111, 113],
               "city": ["서울", "부산", "대전", "광주", "서울", "서울", "부산", "대전", "광주", "서울"],
               "birth_year": [1990, 1989, 1992, 1997, 1982, 1991, 1988, 1990, 1995, 1981],
               "name": ["영이", "순돌", "짱구", "태양", "션", "유리", "현아", "태식", "민수", "호식"]}

attri_data_frame1 = DataFrame(attri_data1)
print(pd.cut(attri_data_frame1.ID, 2))

# 0    (99.987, 106.5]
# 1    (99.987, 106.5]
# 2    (99.987, 106.5]
# 3    (99.987, 106.5]
# 4    (99.987, 106.5]
# 5    (99.987, 106.5]
# 6     (106.5, 113.0]
# 7     (106.5, 113.0]
# 8     (106.5, 113.0]
# 9     (106.5, 113.0]
# Name: ID, dtype: category
# Categories (2, interval[float64]): [(99.987, 106.5] < (106.5, 113.0]]

from numpy import nan as NA
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)

df.columns=["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash",
            "Magnesium", "Total phenols", "Flavanoids",
            "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue",
            "OD280/OD315 of diluted wines","Proline"]

df_ten = df.head(10)
print(df_ten)

df_ten.iloc[1,0] = NA
df_ten.iloc[2,3] = NA
df_ten.iloc[4,8] = NA
df_ten.iloc[7,3] = NA
print(df_ten)

df_ten.fillna(df_ten.mean())
print(df_ten)

print(df_ten["Alcohol"].mean())

df_ten.append(df_ten.loc[3])
df_ten.append(df_ten.loc[6])
df_ten.append(df_ten.loc[9])
df_ten = df_ten.drop_duplicates()
print(df_ten)

alcohol_bins = [0,5,10,15,20,25]
alcoholr_cut_data = pd.cut(df_ten["Alcohol"],alcohol_bins)

print(pd.value_counts(alcoholr_cut_data))


#       Alcohol  Malic acid   Ash  Alcalinity of ash  Magnesium  Total phenols  Flavanoids  Nonflavanoid phenols  Proanthocyanins  Color intensity   Hue  OD280/OD315 of diluted wines  Proline
# 0  1    14.23        1.71  2.43               15.6        127           2.80        3.06                  0.28             2.29             5.64  1.04                          3.92     1065
# 1  1    13.20        1.78  2.14               11.2        100           2.65        2.76                  0.26             1.28             4.38  1.05                          3.40     1050
# 2  1    13.16        2.36  2.67               18.6        101           2.80        3.24                  0.30             2.81             5.68  1.03                          3.17     1185
# 3  1    14.37        1.95  2.50               16.8        113           3.85        3.49                  0.24             2.18             7.80  0.86                          3.45     1480
# 4  1    13.24        2.59  2.87               21.0        118           2.80        2.69                  0.39             1.82             4.32  1.04                          2.93      735
# 5  1    14.20        1.76  2.45               15.2        112           3.27        3.39                  0.34             1.97             6.75  1.05                          2.85     1450
# 6  1    14.39        1.87  2.45               14.6         96           2.50        2.52                  0.30             1.98             5.25  1.02                          3.58     1290
# 7  1    14.06        2.15  2.61               17.6        121           2.60        2.51                  0.31             1.25             5.05  1.06                          3.58     1295
# 8  1    14.83        1.64  2.17               14.0         97           2.80        2.98                  0.29             1.98             5.20  1.08                          2.85     1045
# 9  1    13.86        1.35  2.27               16.0         98           2.98        3.15                  0.22             1.85             7.22  1.01                          3.55     1045
# C:\Users\Inerial\anaconda3\lib\site-packages\pandas\core\indexing.py:965: SettingWithCopyWarning:
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead

# See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
#   self.obj[item] = s
#         Alcohol  Malic acid   Ash  Alcalinity of ash  Magnesium  Total phenols  Flavanoids  Nonflavanoid phenols  Proanthocyanins  Color intensity   Hue  OD280/OD315 of diluted wines  Proline
# 0  1.0    14.23        1.71  2.43               15.6        127           2.80        3.06                  0.28             2.29             5.64  1.04                          3.92     1065
# 1  NaN    13.20        1.78  2.14               11.2        100           2.65        2.76                  0.26             1.28             4.38  1.05                          3.40     1050
# 2  1.0    13.16        2.36   NaN               18.6        101           2.80        3.24                  0.30             2.81             5.68  1.03                          3.17     1185
# 3  1.0    14.37        1.95  2.50               16.8        113           3.85        3.49                  0.24             2.18             7.80  0.86                          3.45     1480
# 4  1.0    13.24        2.59  2.87               21.0        118           2.80        2.69                   NaN             1.82             4.32  1.04                          2.93      735
# 5  1.0    14.20        1.76  2.45               15.2        112           3.27        3.39                  0.34             1.97             6.75  1.05                          2.85     1450
# 6  1.0    14.39        1.87  2.45               14.6         96           2.50        2.52                  0.30             1.98             5.25  1.02                          3.58     1290
# 7  1.0    14.06        2.15   NaN               17.6        121           2.60        2.51                  0.31             1.25             5.05  1.06                          3.58     1295
# 8  1.0    14.83        1.64  2.17               14.0         97           2.80        2.98                  0.29             1.98             5.20  1.08                          2.85     1045
# 9  1.0    13.86        1.35  2.27               16.0         98           2.98        3.15                  0.22             1.85             7.22  1.01                          3.55     1045
#         Alcohol  Malic acid   Ash  Alcalinity of ash  Magnesium  Total phenols  Flavanoids  Nonflavanoid phenols  Proanthocyanins  Color intensity   Hue  OD280/OD315 of diluted wines  Proline
# 0  1.0    14.23        1.71  2.43               15.6        127           2.80        3.06                  0.28             2.29             5.64  1.04                          3.92     1065
# 1  NaN    13.20        1.78  2.14               11.2        100           2.65        2.76                  0.26             1.28             4.38  1.05                          3.40     1050
# 2  1.0    13.16        2.36   NaN               18.6        101           2.80        3.24                  0.30             2.81             5.68  1.03                          3.17     1185
# 3  1.0    14.37        1.95  2.50               16.8        113           3.85        3.49                  0.24             2.18             7.80  0.86                          3.45     1480
# 4  1.0    13.24        2.59  2.87               21.0        118           2.80        2.69                   NaN             1.82             4.32  1.04                          2.93      735
# 5  1.0    14.20        1.76  2.45               15.2        112           3.27        3.39                  0.34             1.97             6.75  1.05                          2.85     1450
# 6  1.0    14.39        1.87  2.45               14.6         96           2.50        2.52                  0.30             1.98             5.25  1.02                          3.58     1290
# 7  1.0    14.06        2.15   NaN               17.6        121           2.60        2.51                  0.31             1.25             5.05  1.06                          3.58     1295
# 8  1.0    14.83        1.64  2.17               14.0         97           2.80        2.98                  0.29             1.98             5.20  1.08                          2.85     1045
# 9  1.0    13.86        1.35  2.27               16.0         98           2.98        3.15                  0.22             1.85             7.22  1.01                          3.55     1045
# 13.954000000000002
#         Alcohol  Malic acid   Ash  Alcalinity of ash  Magnesium  Total phenols  Flavanoids  Nonflavanoid phenols  Proanthocyanins  Color intensity   Hue  OD280/OD315 of diluted wines  Proline
# 0  1.0    14.23        1.71  2.43               15.6        127           2.80        3.06                  0.28             2.29             5.64  1.04                          3.92     1065
# 1  NaN    13.20        1.78  2.14               11.2        100           2.65        2.76                  0.26             1.28             4.38  1.05                          3.40     1050
# 2  1.0    13.16        2.36   NaN               18.6        101           2.80        3.24                  0.30             2.81             5.68  1.03                          3.17     1185
# 3  1.0    14.37        1.95  2.50               16.8        113           3.85        3.49                  0.24             2.18             7.80  0.86                          3.45     1480
# 4  1.0    13.24        2.59  2.87               21.0        118           2.80        2.69                   NaN             1.82             4.32  1.04                          2.93      735
# 5  1.0    14.20        1.76  2.45               15.2        112           3.27        3.39                  0.34             1.97             6.75  1.05                          2.85     1450
# 6  1.0    14.39        1.87  2.45               14.6         96           2.50        2.52                  0.30             1.98             5.25  1.02                          3.58     1290
# 7  1.0    14.06        2.15   NaN               17.6        121           2.60        2.51                  0.31             1.25             5.05  1.06                          3.58     1295
# 8  1.0    14.83        1.64  2.17               14.0         97           2.80        2.98                  0.29             1.98             5.20  1.08                          2.85     1045
# 9  1.0    13.86        1.35  2.27               16.0         98           2.98        3.15                  0.22             1.85             7.22  1.01                          3.55     1045
# (10, 15]    10
# (20, 25]     0
# (15, 20]     0
# (5, 10]      0
# (0, 5]       0
# Name: Alcohol, dtype: int64
import pandas as pd

fruits = {"banana": 3, "orange": 2}
print(pd.Series(fruits))
# banana    3
# orange    2
# dtype: int64




data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}
df = pd.DataFrame(data)
print(df)
#        fruits  year  time
# 0       apple  2001     1
# 1      orange  2002     4
# 2      banana  2001     5
# 3  strawberry  2008     6
# 4   kiwifruit  2006     3




index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)
data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}

df = pd.DataFrame(data)
print("Series 데이터")
print(series)
print("\n")
print("DataFrame 데이터")
print(df)
# Series 데이터
# apple         10
# orange         5
# banana         8
# strawberry    12
# kiwifruit      3
# dtype: int64


# DataFrame 데이터
#        fruits  year  time
# 0       apple  2001     1
# 1      orange  2002     4
# 2      banana  2001     5
# 3  strawberry  2008     6
# 4   kiwifruit  2006     3

fruits = {"banana": 3, "orange": 2}
print(pd.Series(fruits))
# banana    3
# orange    2
# dtype: int64




index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]

series = pd.Series(data, index=index)
print(series)
# apple         10
# orange         5
# banana         8
# strawberry    12
# kiwifruit      3
# dtype: int64



fruits = {"banana": 3, "orange": 4, "grape": 1, "peach": 5}
series = pd.Series(fruits)
print(series[0:2])
# banana    3
# grape     1
# dtype: int64


print(series[["orange", "peach"]])
# orange    4
# peach     5
# dtype: int64




index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)
items1 = series[1:4]

items2 = series[["apple", "banana", "kiwifruit"]]
print(items1)
print()
print(items2)
# orange         5
# banana         8
# strawberry    12
# dtype: int64

# apple        10
# banana        8
# kiwifruit     3
# dtype: int64



index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)
series_values = series.values
series_index = series.index

print(series_values)
print(series_index)
# [10  5  8 12  3]
# Index(['apple', 'orange', 'banana', 'strawberry', 'kiwifruit'], dtype='object')




index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)

pineapple = pd.Series([12], index=["pineapple"])
series = series.append(pineapple)
print(series)

# apple         10
# orange         5
# banana         8
# strawberry    12
# kiwifruit      3
# pineapple     12
# dtype: int64






index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)

series = series.drop("strawberry")

print(series)

# apple        10
# orange        5
# banana        8
# kiwifruit     3
# dtype: int64



index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)

conditions = [True, True, False, False, False]
print(series[conditions])
# apple     10
# orange     5
# dtype: int64




print(series[series >= 5])

# apple         10
# orange         5
# banana         8
# strawberry    12
# dtype: int64





index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)

series = series[series >= 5][series < 10]

print(series)
# orange    5
# banana    8
# dtype: int64



index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data = [10, 5, 8, 12, 3]
series = pd.Series(data, index=index)

items1 = series.sort_index()

items2 = series.sort_values()

print(items1)
print()
print(items2)

# apple         10
# banana         8
# kiwifruit      3
# orange         5
# strawberry    12
# dtype: int64

# kiwifruit      3
# orange         5
# banana         8
# apple         10
# strawberry    12
# dtype: int64






data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}
df = pd.DataFrame(data)
print(df)


#        fruits  year  time
# 0       apple  2001     1
# 1      orange  2002     4
# 2      banana  2001     5
# 3  strawberry  2008     6
# 4   kiwifruit  2006     3







index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data1 = [10, 5, 8, 12, 3]
data2 = [30, 25, 12, 10, 8]
series1 = pd.Series(data1, index=index)
series2 = pd.Series(data2, index=index)

df = pd.DataFrame([series1, series2])

print(df)

#    apple  orange  banana  strawberry  kiwifruit
# 0     10       5       8          12          3
# 1     30      25      12          10          8





data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}
df = pd.DataFrame(data)
series = pd.Series(["mango", 2008, 7], index=["fruits", "year", "time"])
df = df.append(series, ignore_index=True)
print(df)

#        fruits  year  time
# 0       apple  2001     1
# 1      orange  2002     4
# 2      banana  2001     5
# 3  strawberry  2008     6
# 4   kiwifruit  2006     3
# 5       mango  2008     7




index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data1 = [10, 5, 8, 12, 3]
data2 = [30, 25, 12, 10, 8]
data3 = [30, 12, 10, 8, 25, 3]
series1 = pd.Series(data1, index=index)
series2 = pd.Series(data2, index=index)

index.append("pineapple")
series3 = pd.Series(data3, index=index)
df = pd.DataFrame([series1, series2])

df = df.append(series3, ignore_index=True)
print(df)


#    apple  orange  banana  strawberry  kiwifruit  pineapple
# 0     10       5       8          12          3        NaN
# 1     30      25      12          10          8        NaN
# 2     30      12      10           8         25        3.0


data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}
df = pd.DataFrame(data)

df["price"] = [150, 120, 100, 300, 150]
print(df)
#        fruits  year  time  price
# 0       apple  2001     1    150
# 1      orange  2002     4    120
# 2      banana  2001     5    100
# 3  strawberry  2008     6    300
# 4   kiwifruit  2006     3    150








index = ["apple", "orange", "banana", "strawberry", "kiwifruit"]
data1 = [10, 5, 8, 12, 3]
data2 = [30, 25, 12, 10, 8]
series1 = pd.Series(data1, index=index)
series2 = pd.Series(data2, index=index)

new_column = pd.Series([15, 7], index=[0, 1])

df = pd.DataFrame([series1, series2])
df["mango"] = new_column

print(df)
#    apple  orange  banana  strawberry  kiwifruit  mango
# 0     10       5       8          12          3     15
# 1     30      25      12          10          8      7





data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}
df = pd.DataFrame(data)

print(df)
#        fruits  year  time
# 0       apple  2001     1
# 1      orange  2002     4
# 2      banana  2001     5
# 3  strawberry  2008     6
# 4   kiwifruit  2006     3





df = df.loc[[1,2],["time","year"]]
print(df)

#    time  year
# 1     4  2002
# 2     5  2001




import numpy as np
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
    
df.index = range(1, 11)
df = df.loc[range(2,6),["banana","kiwifruit"]]
print(df)
#    banana  kiwifruit
# 2      10         10
# 3       9          1
# 4      10          5
# 5       5          8






data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}

df = pd.DataFrame(data)
print(df)
#        fruits  year  time
# 0       apple  2001     1
# 1      orange  2002     4
# 2      banana  2001     5
# 3  strawberry  2008     6
# 4   kiwifruit  2006     3





df = df.iloc[[1, 3], [0, 2]]
print(df)
#        fruits  time
# 1      orange     4
# 3  strawberry     6



np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

df = df.iloc[range(1,5), [2, 4]]

print(df)
#    banana  kiwifruit
# 2      10         10
# 3       9          1
# 4      10          5
# 5       5          8




data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "time": [1, 4, 5, 6, 3],
        "year": [2001, 2002, 2001, 2008, 2006]}
df = pd.DataFrame(data)

df_1 = df.drop(range(0, 2))

df_2 = df.drop("year", axis=1)

print(df_1)
print()
print(df_2)
#        fruits  time  year
# 2      banana     5  2001
# 3  strawberry     6  2008
# 4   kiwifruit     3  2006

#        fruits  time
# 0       apple     1
# 1      orange     4
# 2      banana     5
# 3  strawberry     6
# 4   kiwifruit     3




np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

df = df.drop(np.arange(2, 11, 2))
df = df.drop("strawberry", axis=1) 

print(df)
#    apple  orange  banana  kiwifruit
# 1      6       8       6         10
# 3      4       9       9          1
# 5      8       2       5          8
# 7      4       8       1          3
# 9      3       9       6          3




data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "time": [1, 4, 5, 6, 3],
        "year": [2001, 2002, 2001, 2008, 2006]}
df = pd.DataFrame(data)
print(df)

df = df.sort_values(by="year", ascending = True)
print(df)

df = df.sort_values(by=["time", "year"] , ascending = True)
print(df)
#        fruits  time  year
# 0       apple     1  2001
# 1      orange     4  2002
# 2      banana     5  2001
# 3  strawberry     6  2008
# 4   kiwifruit     3  2006
#        fruits  time  year
# 0       apple     1  2001
# 2      banana     5  2001
# 1      orange     4  2002
# 4   kiwifruit     3  2006
# 3  strawberry     6  2008
#        fruits  time  year
# 0       apple     1  2001
# 4   kiwifruit     3  2006
# 1      orange     4  2002
# 2      banana     5  2001
# 3  strawberry     6  2008






np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

df = df.sort_values(by=columns)

print(df)

#     apple  orange  banana  strawberry  kiwifruit
# 2       1       7      10           4         10
# 9       3       9       6           1          3
# 7       4       8       1           4          3
# 3       4       9       9           9          1
# 4       4       9      10           2          5
# 10      5       2       1           2          1
# 8       6       8       4           8          8
# 1       6       8       6           3         10
# 5       8       2       5           4          8
# 6      10       7       4           4          4



data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}
df = pd.DataFrame(data)
print(df.index % 2 == 0)
print()
print(df[df.index % 2 == 0])

# [ True False  True False  True]

#       fruits  year  time
# 0      apple  2001     1
# 2     banana  2001     5
# 4  kiwifruit  2006     3






np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

df = df.loc[df["apple"] >= 5]
df = df.loc[df["kiwifruit"] >= 5]

print(df)

#    apple  orange  banana  strawberry  kiwifruit
# 1      6       8       6           3         10
# 5      8       2       5           4          8
# 8      6       8       4           8          8






index = ["growth", "mission", "ishikawa", "pro"]
data = [50, 7, 26, 1]

series = pd.Series(data, index=index)

aidemy = series.sort_index()

aidemy1 = pd.Series([30], index=["tutor"])
aidemy2 = series.append(aidemy1)

print(aidemy)
print()
print(aidemy2)

df = pd.DataFrame()
for index in index:
    df[index] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

aidemy3 = df.loc[range(2,6),["ishikawa"]]
print()
print(aidemy3)

# growth      50
# ishikawa    26
# mission      7
# pro          1
# dtype: int64

# growth      50
# mission      7
# ishikawa    26
# pro          1
# tutor       30
# dtype: int64

#    ishikawa
# 2         4
# 3         6
# 4        10
# 5         5








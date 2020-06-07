import numpy as np
import pandas as pd

def make_random_df(index, columns, seed):
    np.random.seed(seed)
    df = pd.DataFrame()
    for column in columns:
        df[column] = np.random.choice(range(1, 101), len(index))
    df.index = index
    return df

columns = ["apple", "orange", "banana"]
df_data1 = make_random_df(range(1, 5), columns, 0)
df_data2 = make_random_df(range(1, 5), columns, 1)

df1 = pd.concat([df_data1, df_data2], axis=0)
df2 = pd.concat([df_data1, df_data2], axis=1)

print(df1)
print(df2)

#    apple  orange  banana
# 1     45      68      37
# 2     48      10      88
# 3     65      84      71
# 4     68      22      89
# 1     38      76      17
# 2     13       6       2
# 3     73      80      77
# 4     10      65      72
#    apple  orange  banana  apple  orange  banana
# 1     45      68      37     38      76      17
# 2     48      10      88     13       6       2
# 3     65      84      71     73      80      77
# 4     68      22      89     10      65      72







def make_random_df(index, columns, seed):
    np.random.seed(seed)
    df = pd.DataFrame()
    for column in columns:
        df[column] = np.random.choice(range(1, 101), len(index))
    df.index = index
    return df

columns1 = ["apple", "orange", "banana"]
columns2 = ["orange", "kiwifruit", "banana"]

df_data1 = make_random_df(range(1, 5), columns1, 0)
df_data2 = make_random_df(np.arange(1, 8, 2), columns2, 1)
df1 = pd.concat([df_data1, df_data2], axis=0)
df2 = pd.concat([df_data1, df_data2], axis=1) 

print(df1)
print(df2)
#    apple  orange  banana  kiwifruit
# 1   45.0      68      37        NaN
# 2   48.0      10      88        NaN
# 3   65.0      84      71        NaN
# 4   68.0      22      89        NaN
# 1    NaN      38      17       76.0
# 3    NaN      13       2        6.0
# 5    NaN      73      77       80.0
# 7    NaN      10      72       65.0
#    apple  orange  banana  orange  kiwifruit  banana
# 1   45.0    68.0    37.0    38.0       76.0    17.0
# 2   48.0    10.0    88.0     NaN        NaN     NaN
# 3   65.0    84.0    71.0    13.0        6.0     2.0
# 4   68.0    22.0    89.0     NaN        NaN     NaN
# 5    NaN     NaN     NaN    73.0       80.0    77.0
# 7    NaN     NaN     NaN    10.0       65.0    72.0






def make_random_df(index, columns, seed):
    np.random.seed(seed)
    df = pd.DataFrame()
    for column in columns:
            df[column] = np.random.choice(range(1, 101), len(index))
    df.index = index
    return df

columns = ["apple", "orange", "banana"]
df_data1 = make_random_df(range(1, 5), columns, 0)
df_data2 = make_random_df(range(1, 5), columns, 1)

df = pd.concat([df_data1, df_data2], axis=1, keys=["X", "Y"])

Y_banana = df["Y", "banana"]

print(df)
print()
print(Y_banana)
#       X                   Y
#   apple orange banana apple orange banana
# 1    45     68     37    38     76     17
# 2    48     10     88    13      6      2
# 3    65     84     71    73     80     77
# 4    68     22     89    10     65     72

# 1    17
# 2     2
# 3    77
# 4    72
# Name: (Y, banana), dtype: int32








data1 = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
         "year": [2001, 2002, 2001, 2008, 2006],
         "amount": [1, 4, 5, 6, 3]}
df1 = pd.DataFrame(data1)

data2 = {"fruits": ["apple", "orange", "banana", "strawberry", "mango"],
         "year": [2001, 2002, 2001, 2008, 2007],
         "price": [150, 120, 100, 250, 3000]}
df2 = pd.DataFrame(data2)

print(df1)
print()
print(df2)
print()
df3 = pd.merge(df1, df2, on="fruits", how="inner")

print(df3)
#        fruits  year  amount
# 0       apple  2001       1
# 1      orange  2002       4
# 2      banana  2001       5
# 3  strawberry  2008       6
# 4   kiwifruit  2006       3

#        fruits  year  price
# 0       apple  2001    150
# 1      orange  2002    120
# 2      banana  2001    100
# 3  strawberry  2008    250
# 4       mango  2007   3000

#        fruits  year_x  amount  year_y  price
# 0       apple    2001       1    2001    150
# 1      orange    2002       4    2002    120
# 2      banana    2001       5    2001    100
# 3  strawberry    2008       6    2008    250








data1 = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
         "year": [2001, 2002, 2001, 2008, 2006],
         "amount": [1, 4, 5, 6, 3]}

df1 = pd.DataFrame(data1)

data2 = {"fruits": ["apple", "orange", "banana", "strawberry", "mango"],
         "year": [2001, 2002, 2001, 2008, 2007],
         "price": [150, 120, 100, 250, 3000]}
df2 = pd.DataFrame(data2)

print(df1)
print()
print(df2)
print()

df3 = pd.merge(df1, df2, on="fruits", how="outer")

print(df3)

#        fruits  year  amount
# 0       apple  2001       1
# 1      orange  2002       4
# 2      banana  2001       5
# 3  strawberry  2008       6
# 4   kiwifruit  2006       3

#        fruits  year  price
# 0       apple  2001    150
# 1      orange  2002    120
# 2      banana  2001    100
# 3  strawberry  2008    250
# 4       mango  2007   3000

#        fruits  year_x  amount  year_y   price
# 0       apple  2001.0     1.0  2001.0   150.0
# 1      orange  2002.0     4.0  2002.0   120.0
# 2      banana  2001.0     5.0  2001.0   100.0
# 3  strawberry  2008.0     6.0  2008.0   250.0
# 4   kiwifruit  2006.0     3.0     NaN     NaN
# 5       mango     NaN     NaN  2007.0  3000.0






order_df = pd.DataFrame([[1000, 2546, 103],
                         [1001, 4352, 101],
                         [1002, 342, 101]],
                        columns=["id", "item_id", "customer_id"])

customer_df = pd.DataFrame([[101, "광수"],
                            [102, "민호"],
                            [103, "소희"]],
                           columns=["id", "name"])

order_df = pd.merge(order_df, customer_df, left_on="customer_id", right_on="id", how="inner")

print(order_df)

#    id_x  item_id  customer_id  id_y name
# 0  1000     2546          103   103   소희
# 1  1001     4352          101   101   광수
# 2  1002      342          101   101   광수






order_df = pd.DataFrame([[1000, 2546, 103],
                         [1001, 4352, 101],
                         [1002, 342, 101]],
                        columns=["id", "item_id", "customer_id"])

customer_df = pd.DataFrame([["광수"],
                            ["민호"],
                            ["소희"]],
                           columns=["name"])
customer_df.index = [101, 102, 103]

order_df = pd.merge(order_df, customer_df, left_on="customer_id", right_index=True, how="inner")

print(order_df)

#      id  item_id  customer_id name
# 0  1000     2546          103   소희
# 1  1001     4352          101   광수
# 2  1002      342          101   광수







np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)
df_head = df.head(3)
df_tail = df.tail(3)

print(df_head)
print(df_tail)

#    apple  orange  banana  strawberry  kiwifruit
# 1      6       8       6           3         10
# 2      1       7      10           4         10
# 3      4       9       9           9          1
#     apple  orange  banana  strawberry  kiwifruit
# 8       6       8       4           8          8
# 9       3       9       6           1          3
# 10      5       2       1           2          1






import math
np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

double_df = df * 2
square_df = df * df
sqrt_df = np.sqrt(df) 

print(double_df)
print(square_df)
print(sqrt_df)
#     apple  orange  banana  strawberry  kiwifruit
# 1      12      16      12           6         20
# 2       2      14      20           8         20
# 3       8      18      18          18          2
# 4       8      18      20           4         10
# 5      16       4      10           8         16
# 6      20      14       8           8          8
# 7       8      16       2           8          6
# 8      12      16       8          16         16
# 9       6      18      12           2          6
# 10     10       4       2           4          2
#     apple  orange  banana  strawberry  kiwifruit
# 1      36      64      36           9        100
# 2       1      49     100          16        100
# 3      16      81      81          81          1
# 4      16      81     100           4         25
# 5      64       4      25          16         64
# 6     100      49      16          16         16
# 7      16      64       1          16          9
# 8      36      64      16          64         64
# 9       9      81      36           1          9
# 10     25       4       1           4          1
#        apple    orange    banana  strawberry  kiwifruit
# 1   2.449490  2.828427  2.449490    1.732051   3.162278
# 2   1.000000  2.645751  3.162278    2.000000   3.162278
# 3   2.000000  3.000000  3.000000    3.000000   1.000000
# 4   2.000000  3.000000  3.162278    1.414214   2.236068
# 5   2.828427  1.414214  2.236068    2.000000   2.828427
# 6   3.162278  2.645751  2.000000    2.000000   2.000000
# 7   2.000000  2.828427  1.000000    2.000000   1.732051
# 8   2.449490  2.828427  2.000000    2.828427   2.828427
# 9   1.732051  3.000000  2.449490    1.000000   1.732051
# 10  2.236068  1.414214  1.000000    1.414214   1.000000





np.random.seed(0)
columns = ["apple", "orange", "banana", "strawberry", "kiwifruit"]

df = pd.DataFrame()
for column in columns:
    df[column] = np.random.choice(range(1, 11), 10)
df.index = range(1, 11)

df_des = df.describe().loc[["mean", "max", "min"]]

print(df_des)
#       apple  orange  banana  strawberry  kiwifruit
# mean    5.1     6.9     5.6         4.1        5.3
# max    10.0     9.0    10.0         9.0       10.0
# min     1.0     2.0     1.0         1.0        1.0







prefecture_df = pd.DataFrame([["강릉", 2190, 13636, "강원도"], 
                              ["광주", 2415, 9145, "전라도"],
                              ["평창", 1904, 8837, "강원도"],
                              ["대전", 4610, 2605, "충청도"],
                              ["단양", 5172, 7505, "충청도"]],
                             columns=["Prefecture", "Area",
                                      "Population", "Region"])

print(prefecture_df)

grouped_region = prefecture_df.groupby("Region")

mean_df = grouped_region.mean()

print(mean_df)
#   Prefecture  Area  Population Region
# 0         강릉  2190       13636    강원도
# 1         광주  2415        9145    전라도
# 2         평창  1904        8837    강원도
# 3         대전  4610        2605    충청도
# 4         단양  5172        7505    충청도
#           Area  Population
# Region
# 강원도     2047.0     11236.5
# 전라도     2415.0      9145.0
# 충청도     4891.0      5055.0





df1 = pd.DataFrame([["apple", "Fruit", 120],
                    ["orange", "Fruit", 60],
                    ["banana", "Fruit", 100],
                    ["pumpkin", "Vegetable", 150],
                    ["potato", "Vegetable", 80]],
                    columns=["Name", "Type", "Price"])

df2 = pd.DataFrame([["onion", "Vegetable", 60],
                    ["carrot", "Vegetable", 50],
                    ["beans", "Vegetable", 100],
                    ["grape", "Fruit", 160],
                    ["kiwifruit", "Fruit", 80]],
                    columns=["Name", "Type", "Price"])

df3 = pd.concat([df1, df2], axis=0)

df_fruit = df3.loc[df3["Type"] == "Fruit"]
df_fruit = df_fruit.sort_values(by="Price")

df_veg = df3.loc[df3["Type"] == "Vegetable"]
df_veg = df_veg.sort_values(by="Price")

print(sum(df_fruit[:3]["Price"]) + sum(df_veg[:3]["Price"]))

# 430





index = ["광수", "민호", "소희", "태양", "영희"]
columns = ["국어", "수학", "사회", "과학", "영어"]
data = [[30, 45, 12, 45, 87], [65, 47, 83, 17, 58], [64, 63, 86, 57, 46,], [38, 47, 62, 91, 63], [65, 36, 85, 94, 36]]
df = pd.DataFrame(data, index=index, columns=columns)

pe_column = pd.Series([56, 43, 73, 82, 62], index=["광수", "민호", "소희", "태양", "영희"])
df["체육"] = pe_column
print(df)
print()

df1 = df.sort_values(by="수학", ascending=True)
print(df1)
print()

df2 = df1 + 5
print(df2)
print()

print(df2.describe().loc[["mean", "max", "min"]])

#     국어  수학  사회  과학  영어  체육
# 광수  30  45  12  45  87  56
# 민호  65  47  83  17  58  43
# 소희  64  63  86  57  46  73
# 태양  38  47  62  91  63  82
# 영희  65  36  85  94  36  62

#     국어  수학  사회  과학  영어  체육
# 영희  65  36  85  94  36  62
# 광수  30  45  12  45  87  56
# 민호  65  47  83  17  58  43
# 태양  38  47  62  91  63  82
# 소희  64  63  86  57  46  73

# 영희  70  41  90  99  41  67
# 광수  35  50  17  50  92  61
# 민호  70  52  88  22  63  48
# 태양  43  52  67  96  68  87
# 소희  69  68  91  62  51  78

#         국어    수학    사회    과학    영어    체육
# mean  57.4  52.6  70.6  65.8  63.0  68.2
# max   70.0  68.0  91.0  99.0  92.0  87.0
# min   35.0  41.0  17.0  22.0  41.0  48.0
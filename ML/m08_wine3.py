import pandas as pd
import matplotlib.pyplot as plt

## 1. 데이터
wine = pd.read_csv('./ml_prac/csv/winequality-white.csv', sep=';',
                   header = 0, index_col = None)

count_data = wine.groupby('quality')['quality'].count()

print(count_data)

count_data.plot()
plt.show()
## 데이터가 가운데에 몰려있음
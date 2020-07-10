import pandas as pd
import numpy as np
from keras.layers import Dense, LSTM, Dropout, Input
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

kospi200 = pd.read_csv('./data/csv/kospi200.csv', header=0, index_col=0, sep=',', encoding = 'cp949')
samsung = pd.read_csv('./data/csv/samsung.csv', header=0, index_col=0, sep=',', encoding = 'cp949')

print(samsung.sort_values(by = ['일자'], ascending=[True]))

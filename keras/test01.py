import pandas as pd
import numpy as np

import seaborn as sns 
import matplotlib.pyplot as plt 

x = pd.read_csv('./data/dacon/comp3/train_features.csv', sep=',', index_col = None, header = 0)
y = pd.read_csv('./data/dacon/comp3/train_target.csv', sep=',', index_col = 0, header = 0)
x_pred = pd.read_csv('./data/dacon/comp3/test_features.csv', sep=',', index_col = None, header = 0)

def vibration_show(df, idx):
    # df[df.id==idx].S1.plot()
    # df[df.id==idx].S2.plot()
    df[df.id==idx].S3.plot()
    df[df.id==idx].S4.plot()

for i in range(10):
    vibration_show(x, i)
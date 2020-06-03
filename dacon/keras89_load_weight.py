import numpy as np
import pandas as pd
import os

filep = os.path.dirname(os.path.realpath(__file__))
test = pd.read_csv(filep + '/csv/test.csv',index_col = 0, header = 0, sep=',')
train = pd.read_csv(filep + '/csv/train.csv',index_col = 0, header = 0, sep=',')

print(test.columns)
print(train.columns)
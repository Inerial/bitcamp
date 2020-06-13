from konlpy.tag import Kkma
from konlpy.utils import pprint
import numpy as np
import os
import pandas as pd

ko = Kkma()
# print(np.array(ko.pos("10연속 사진 촬영")))
# print(np.array(ko.pos("사진 열번 찍어")))
# print(np.array(ko.pos("5번 찍어주지 않을래?"))) ## 단락별로 나누고 단어 특성
# print(np.array(ko.morphs("5번 찍어주지 않을래?"))) ## 단락별로 나누고 단어 특성

# input() # 입력받는 함수

pypath = os.path.dirname(os.path.realpath(__file__))

fp = open(file=pypath+'\mykakao.txt', mode='rt', encoding='utf-8')
text1 = np.array([i.replace('\n', '') for i in fp.readlines()])
text1 = np.delete(text1, np.where(text1 == ''))
fp.close()

## 대화에서 일정개수 랜덤추출하여 단어특성으로 분류
x_train = []
y_train = []
for i in range(20):
    x_train.append(ko.morphs(text1[np.random.randint(text1.shape[0]-1)]))
    y_train.append(0)

train = pd.read_csv(pypath+'//train_chatdata.csv', sep=',', index_col=None, header=None).values
for i in range(train.shape[0]):
    x_train.append(ko.morphs(train[i][0]))
    y_train.append(1)

print(np.array(x_train).shape)
print(np.array(y_train).shape)

word2index = {'<unk>':0}
for x in x_train:
    for token in x:
        if word2index.get(token)==None:
            word2index[token]= len(word2index)

class2indxe = {'사진': 1, "없음" : 0}
print(word2index)

def make_Bow(seq, word2index):
    tensor = np.zeros(len(word2index))
    for w in seq:
        index = word2index.get(w)
        if index != None:
            tensor[index] += 1
        else:
            index = word2index['<unk>']
            tensor[index] += 1
    return tensor

# train.
from konlpy.tag import Kkma
from konlpy.utils import pprint
import numpy as np
import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier

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
    y_train.append([np.nan])

class2index = {"<unk>" : 0}
train = pd.read_csv(pypath+'//train_chatdata.csv', sep=',', index_col=None, header=None).values
for i in range(train.shape[0]):
    x_train.append(ko.morphs(train[i][0]))
    y_train.append(train[i][1:])
    for token in train[i][1:]:
        if type(token) == float:
            break
        if class2index.get(token)==None:
            class2index[token] = len(class2index)

print(np.array(x_train).shape)
print(np.array(y_train).shape)

word2index = {'<unk>':0}
for x in x_train:
    for token in x:
        if word2index.get(token)==None:
            word2index[token]= len(word2index)

print(class2index)

## 각 문장에 들어있는 단어들로 one hot encoding
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

x_train = np.array([make_Bow(x,word2index) for x in x_train])
y_train = np.array([make_Bow(y,class2index) for y in y_train])
print(x_train.shape)
print(y_train.shape)

model = XGBClassifier()
model = MultiOutputClassifier(model)
model.fit(x_train , y_train[:,1:])


while True:
    print("종료 : 종료 입력시")
    text = input()
    if text == "종료":
        break
    x_pred = model.predict(np.array([make_Bow(ko.morphs(text), word2index)]))
    print(x_pred)
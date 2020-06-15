from konlpy.tag import Kkma
from konlpy.utils import pprint
import numpy as np
import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
import nltk
import torch
import torch.nn as nn

ko = Kkma()

pypath = os.path.dirname(os.path.realpath(__file__))

fp = open(file=pypath+'\mykakao.txt', mode='rt', encoding='utf-8')
text1 = np.array([i.replace('\n', '') for i in fp.readlines()])
text1 = np.delete(text1, np.where(text1 == ''))
fp.close()

## 대화에서 일정개수 랜덤추출하여 단어특성으로 분류
x_tokens = []

for i in text1:
    x_tokens.append(list(ko.morphs(i)))

# print(x_tokens)

word2index={}
for vocabulary in x_tokens:
    for voca in vocabulary:
        if word2index.get(voca)==None:
            word2index[voca]=len(word2index)
# print(word2index)

WINDOW_SIZE = 2
x_windows = []
for tokenized in x_tokens:
    # print(tokenized)
    x_windows.append(list(nltk.ngrams(['<DUMMY>'] * WINDOW_SIZE + tokenized + ['<DUMMY>'] * WINDOW_SIZE, WINDOW_SIZE * 2 + 1)))
print(x_windows)
train_data = []
for windows in x_windows:
    for window in windows:
        for i in range(WINDOW_SIZE * 2 + 1):
            if i == WINDOW_SIZE or window[i] == '<DUMMY>': 
                continue
            train_data.append((window[WINDOW_SIZE], window[i]))

# print(train_data[:30])

# >>> [('I', 'have'), ('I', 'a'), ('have', 'I'), ('have', 'a')]
# 각 단어를 index로 바꾸고 LongTensor로 바꿔주는 함수
def prepare_word(word, word2index):
    return torch.LongTensor([word2index[word]])

X_p,y_p=[],[]

for (center,context) in train_data:
    X_p.append(prepare_word(center, word2index).view(1, -1))
    y_p.append(prepare_word(context, word2index).view(1, -1))
# print(X_p) ## 각각 단어의 word2index에서의 인덱스값
# print(y_p)

train_data = list(zip(X_p,y_p))
# print(train_data[0]) ## 다시 X,y를 묶어줌

center_embed = nn.Embedding(len(word2index),3) ## 각각 단어의 벡터 계산 (함수화 불가능)
context_embed = nn.Embedding(len(word2index),3)
print(center_embed)
print(context_embed)


print(center_embed.weight)
print(context_embed.weight)


center,context = train_data[0]

center_vector = center_embed(center)
context_vector = context_embed(context)
print(center_vector)
print(context_vector)
# 배치 사이즈 : 1 


score = torch.exp(context_vector.bmm(center_vector.transpose(1,2))).squeeze(2)
print('score',score)

# 분모값
#  시퀀스(단어들의 연속된 리스트)가 들어오면 LongTensor로 매핑
def prepare_sequence(seq, word2index):
    idxs = list(map(lambda w: word2index[w], seq))
    return torch.LongTensor(idxs)

vocabulary_tensor = prepare_sequence(vocabulary,word2index).view(1,-1)
print(vocabulary_tensor)

vocabulary_vector = context_embed(vocabulary_tensor)

norm_scores = vocabulary_vector.bmm(center_vector.transpose(1, 2))
print(norm_scores)
norm_scores = torch.sum(torch.exp(norm_scores))
print(norm_scores)

# 결과
print(score/norm_scores)
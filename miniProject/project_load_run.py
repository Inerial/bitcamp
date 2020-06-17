from gensim.models import Doc2Vec, Word2Vec
from gensim.models import KeyedVectors
from konlpy.tag import Mecab
import numpy as np
import os, shutil, pandas as pd , nltk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model, Model
from keras.layers import Input, Dense, Embedding, Flatten
from sklearn.externals import joblib

pypath = os.path.dirname(os.path.realpath(__file__))
ko = Mecab(dicpath='C:\mecab\mecab-ko-dic')

t = joblib.load(pypath+'/tokenizer.pkl')
class2index = np.load(pypath+'/class2index.npy', allow_pickle=True).item()
max_len = class2index["maxlen"]
del class2index["maxlen"]
vocab_size = len(t.word_index)

word_model = KeyedVectors.load_word2vec_format(pypath+'/wiki_dmpv_300_no_taginfo_user_dic_word2vec_format.bin', binary=True)

embedding_matrix = np.zeros((vocab_size, 300))
def get_vector(word):
    if word in word_model:
        return word_model[word]
    else:
        return None
print(t.word_index.items())
for word, i in t.word_index.items():
    temp = get_vector(word)
    if temp is not None:
        embedding_matrix[i-1] = temp


def build_model():
    input1 = Input(shape=(max_len, ))
    dense1 = Embedding(vocab_size, 300, weights=[embedding_matrix])(input1)
    dense1 = Flatten()(dense1)
    output1 = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[input1], outputs=[output1])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model

model = joblib.load(pypath+'/chatModel.pkl')
def re_Bow(seq, index):
    word = []
    for i in range(len(seq)):
        if seq[i] == 1:
            word.append(list(index.keys())[i+1])
    return word

while True:
    print("종료 : 종료 입력시")
    text = input()
    if text == "종료":
        break
    tmp = []
    for token in ko.morphs(text):
        ind = t.word_index.get(token)
        if ind is not None:
            tmp.append(ind)
    tmp = np.array(pad_sequences([tmp], maxlen= max_len, padding='post'))
    
    x_pred = model.predict(tmp)
    print(re_Bow(x_pred[0,0], class2index))

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from konlpy.tag import Mecab
import numpy as np, os, shutil, pandas as pd, nltk, re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Embedding, Flatten, Input,Dropout, LSTM
from sklearn.multioutput import MultiOutputClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.externals import joblib

## Mecab형태소분석이 불러오기, 현재 py파일이 있는 폴더주소 저장
ko = Mecab(dicpath='C:\mecab\mecab-ko-dic')
print(ko.morphs("내가가방에들어간다"))
pypath = os.path.dirname(os.path.realpath(__file__))

## 데이터를 정리해 넣을 빈 list, dict
sentences = []
y_train = []
class2index = {"<unk>" : 0}

## 분석을 위해 입력한 문장 + 문장의 키워드 불러오기
train = pd.read_csv(pypath+'//train_chatdata.csv', sep=',', index_col=None, header=None).values
## 받은 문장 + 키워드 dataframe을 위의 빈 list, dict에 필요한 형태로 넣어주기
for i in range(train.shape[0]):
    text = ''
    for word in ko.morphs(train[i][0]):
        text += word + ' '                  ## Tokenizer가 띄워쓰기를 기준으로 분류
    sentences.append(text)                  ## 따라서 형태소 분석기를 통해 형태소간에 띄워쓰기를 넣어줌

    y_train.append(train[i][1:])            ## 문장옆의 키워드를 분류
    for token in train[i][1:]:
        if type(token) == float:            ## 키워드의 길이가 다르고 빈 공간은 nan값이 들어가있어서 분류
            break
        if class2index.get(token)==None:    ## dict에 키워드가 없다면 idx를 지정해주고 넣어준다.
            class2index[token] = len(class2index)

print(sentences[:3])
print(y_train[:3])
print(class2index)
# ['10 연속 사진 촬영 ', '10 연속 사진 찍 어 ', '사진 10 연속 촬영 ']
# [array(['사진', '10번', '촬영', nan, nan, nan, nan, nan, nan, nan], dtype=object), array(['사진', '10번', '촬영', nan, nan, nan, nan, nan, nan, nan], dtype=object), array(['사진', '10번', '촬영', nan, nan, nan, nan, nan, nan, nan], dtype=object)]
# {'<unk>': 0, '사진': 1, '10번': 2, '촬영': 3, '5번': 4, '2번': 5, '10초': 6, '5초': 7, '2초': 8}


## 위의 y_train값을 
def make_y(seq, word2index):
    tensor = np.zeros(len(word2index))
    for w in seq:
        index = word2index.get(w)
        if index != None:
            tensor[index] += 1
    return tensor
y_train = np.array([make_y(y,class2index)[1:] for y in y_train])

# word2vec 모델 불러오기
word_model = KeyedVectors.load_word2vec_format(pypath+'/wiki_dmpv_300_no_taginfo_user_dic_word2vec_format.bin', binary=True)

t = Tokenizer()
t.fit_on_texts(sentences)
vocab_size = len(t.word_index)  ## 분류된 단어의 종류 수
x_encoded = t.texts_to_sequences(sentences)  ## 단어를 분류한 index로 치환
max_len = max(len(i) for i in x_encoded)  ## 가장 긴 문장의 단어개수

## max_len에 맞춰 0으로 채워넣는 함수  ==  단어간의 길이가 다르기 때문에
x_train = pad_sequences(x_encoded, maxlen= max_len, padding='post')
y_train = np.array(y_train)

embedding_matrix = np.zeros((vocab_size, 300))  ## 사용할 단어들의 벡터값만 꺼내오기

def get_vector(word):  
    if word in word_model:      ## 가져온 word 변수(단어)가 모델안에 있다면
        return word_model[word] ## 가져오기
    else:
        return None

## 분류된 단어들의 Vector값을 다 가져오기
for word, i in t.word_index.items(): 
    temp = get_vector(word)
    if temp is not None:
        embedding_matrix[i-1] = temp

## 모델
def build_model():
    input1 = Input(shape=(max_len, )) ## input 레이어는 LSTM과 다름
    dense1 = Embedding(vocab_size, 300, weights=[embedding_matrix], trainable=True)(input1)
    ##             (단어의 개수, 단어를 표현한 벡터의 column 수,  weight값 미리 지정, 훈련 여부)
    dense1 = LSTM(128, activation='elu')(dense1)
    dense1 = Dropout(0.2)(dense1)
    output1 = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[input1], outputs=[output1])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model

## 각 키워드 별로 각각 분류를 하기 위해 사용
model = KerasClassifier(build_fn=build_model, epochs= 400, batch_size = 100)
model = MultiOutputClassifier(model)
model.fit(x_train,y_train)#, validation_split=0.3)


# input1 = Input(shape=(max_len, )) ## input 레이어는 LSTM과 다름
# dense1 = Embedding(vocab_size, 300, weights=[embedding_matrix], trainable=True)(input1)
# ##             (단어의 개수, 단어를 표현한 벡터의 column 수,  weight값 미리 지정, 훈련 여부)
# dense1 = LSTM(128, activation='elu')(dense1)
# dense1 = Dropout(0.2)(dense1)
# output1 = Dense(8, activation='sigmoid')(dense1)
# model = Model(inputs=[input1], outputs=[output1])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# ## 각 키워드 별로 각각 분류를 하기 위해 사용
# model.fit(x_train,y_train, epochs= 400, batch_size = 100)#, validation_split=0.3)

## 머신러닝 모델을 저장하기 위한 라이브러리 == 나머지 필요한것들도 저장
joblib.dump(model, pypath+'/chatModel.pkl')
joblib.dump(t, pypath+'/tokenizer.pkl')
class2index["maxlen"] = max_len
np.save(pypath+'/class2index.npy', arr=class2index)
del class2index["maxlen"]



## 나온 결과값을 다시 문자로 치환
def re_y(seq, index):
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
    print(re_y(x_pred[0,0], class2index))


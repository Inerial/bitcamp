from gensim.models import Doc2Vec, Word2Vec
from gensim.models import KeyedVectors
from konlpy.tag import Mecab
import numpy as np, os, shutil, pandas as pd, nltk, re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Embedding, Flatten, Input,Dropout
from sklearn.multioutput import MultiOutputClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.externals import joblib

ko = Mecab(dicpath='C:\mecab\mecab-ko-dic')
pypath = os.path.dirname(os.path.realpath(__file__))

sentences = []
y_train = []

class2index = {"<unk>" : 0}
train = pd.read_csv(pypath+'//train_chatdata.csv', sep=',', index_col=None, header=None).values
print(train)
for i in range(train.shape[0]):
    text = ''
    for word in ko.morphs(train[i][0]):
        text += word + ' '
    sentences.append(text)

    y_train.append(train[i][1:])
    for token in train[i][1:]:
        if type(token) == float:
            break
        if class2index.get(token)==None:
            class2index[token] = len(class2index)


def make_y(seq, word2index):
    tensor = np.zeros(len(word2index))
    for w in seq:
        index = word2index.get(w)
        if index != None:
            tensor[index] += 1
    return tensor

y_train = np.array([make_y(y,class2index)[1:] for y in y_train])


word_model = KeyedVectors.load_word2vec_format(pypath+'/wiki_dmpv_300_no_taginfo_user_dic_word2vec_format.bin', binary=True)

print(word_model.similarity("사진", "동영상"))


t = Tokenizer()
t.fit_on_texts(sentences)
vocab_size = len(t.word_index)
print(vocab_size)
x_encoded = t.texts_to_sequences(sentences)
print(x_encoded)

## 가장 긴 문장의 단어개수
max_len = max(len(i) for i in x_encoded)
# print(max_len)

## max_len에 맞춰 0으로 채워넣는 함수
x_train = pad_sequences(x_encoded, maxlen= max_len, padding='post')
y_train = np.array(y_train)
print(x_train)



embedding_matrix = np.zeros((vocab_size, 300))
print(embedding_matrix.shape)

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

## 모델
def build_model():
    input1 = Input(shape=(max_len, ))
    dense1 = Embedding(vocab_size, 300, weights=[embedding_matrix])(input1)
    dense1 = Flatten()(dense1)
    output1 = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[input1], outputs=[output1])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model
model = KerasClassifier(build_fn=build_model, epochs= 400, batch_size = 100)
model = MultiOutputClassifier(model)

model.fit(x_train,y_train)

joblib.dump(model, pypath+'/chatModel.pkl')
joblib.dump(t, pypath+'/tokenizer.pkl')
class2index["maxlen"] = max_len
np.save(pypath+'/class2index.npy', arr=class2index)
del class2index["maxlen"]

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
    print(tmp)
    tmp = np.array(pad_sequences([tmp], maxlen= max_len, padding='post'))
    
    x_pred = model.predict(tmp)
    print(re_Bow(x_pred[0,0], class2index))



## 데이터에 없는 글자들뭉텅이로 묶어 처리할 필요성이 있다.
## 데이터가 길어질수록 인식을 못하는경향이 보인다(데이터의 수가 너무 적나?)
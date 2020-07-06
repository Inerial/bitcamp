from keras.preprocessing.text import Tokenizer
import numpy as np

docs = ["너무 재밌어요", "참 최고에요", "참 잘 만든 영화에요", 
        "추천하고 싶은 영화입니다.", "한번 더 보고 싶네요","글쎄요",
        "별로에요", "생각보다 지루해요", "연기가 어색해요",
        "재미없어요", "너무 재미없다.", "참 재밋네요"]

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)



x = token.texts_to_sequences(docs)
print(x)

from keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', value=0) ## padding = (post, pre),  value= 0
print(pad_x)

from keras.utils import to_categorical

word_size = len(token.word_index) + 1
# pad_x = to_categorical(pad_x, num_classes=word_size)
# print(pad_x)

## 원핫 인코딩 수치를 그대로 가져다가 임베딩에 돌려버린다?
## 수치를 임베딩이 벡터화 해준다?

from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding

model = Sequential()
model.add(Embedding(25, 10, input_length=4)) ## 전체 단어의 수 ,  아웃풋, 전체 단어의 수가 실행에 상관이 없다? 
# model.add(Embedding(25, 10)) ##
## param = word_size * output
model.add(Flatten())
model.add(Dense(1))

model.summary()

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=30)

acc = model.evaluate(pad_x, labels)[1]
print(acc)
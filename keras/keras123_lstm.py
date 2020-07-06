## 122를 그대로 카피해서 123 완성
# embedding을 빼고 lstm으로 완성
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

pad_x = pad_x.reshape(pad_x.shape[0],pad_x.shape[1], 1)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, LSTM, Conv1D

model = Sequential()
model.add(LSTM(10, input_shape=(4,1)))

model.add(Dense(1))

model.summary()

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=30)

acc = model.evaluate(pad_x, labels)[1]
print(acc)

# 12/12 [==============================] - 0s 583us/step - loss: 7.7125 - acc: 0.5000
# Epoch 30/30
# 12/12 [==============================] - 0s 584us/step - loss: 7.7125 - acc: 0.5000
# 12/12 [==============================] - 0s 5ms/step
# 0.5
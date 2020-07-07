from keras.datasets import imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train),(x_test,y_test) = imdb.load_data(num_words=2000)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

print(x_train[0])
print(y_train[0])

print(len(x_train[0]))

# y의 카테고리 개수 출력
category = np.max(y_train)+1
print("카테고리 :", category) # 46

# y의 유니크한 값들 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)

y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()
print(bbb)
print(bbb.shape)

## 그룹별로 개수 저장
## 그룹바이 사용법 알아두기 - 주간과제


from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, maxlen=111, padding='pre')
x_test = pad_sequences(x_test, maxlen=111, padding='pre')

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)

#2. 모델
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten,Dropout, Conv1D, AveragePooling1D, BatchNormalization, ELU
from keras.layers import Bidirectional ## 양방향으로 LSTM
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2

model = Sequential()
model.add(Embedding(2000, 128))

model.add(Conv1D(256,5,padding='valid'))
model.add(BatchNormalization())
model.add(ELU())
model.add(AveragePooling1D(5))
model.add(Dropout(0.2))


model.add(Bidirectional(LSTM(256)))
model.add(BatchNormalization())
model.add(ELU())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(ELU())
model.add(Dense(64))
model.add(BatchNormalization())
model.add(ELU())
model.add(Dense(1, activation='softmax'))

model.summary()

# model.compile(loss='binary_crossentropy', optimizer=Adam(0.0003), metrics=['acc'])
# history = model.fit(x_train, y_train, batch_size=500, epochs=50, validation_split=0.2)

# acc = model.evaluate(x_test, y_test)[1]

# print('acc :', acc)

# y_val_loss = history.history['val_loss']
# y_loss = history.history['loss']

# plt.plot(y_val_loss, marker='.', c='red', label='TestSet Loss')
# plt.plot(y_loss, marker='.', c='blue', label='TrainSet Loss')
# plt.legend(loc='upper right')
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()


# # 1. imdb 검색해 데이터 내용 확인
# # 감정에 따라 (긍정적/부정적)으로 라벨된 25,000개의 IMDB 영화 리뷰로 구성된 데이터셋. 
# # 리뷰는 선행처리되었으며, 각 리뷰는 단어 인덱스(정수)로 구성된 sequence로 인코딩 되었습니다.
# # 편의를 위해 단어는 데이터내 전체적 사용빈도에 따라 인덱스화 되었습니다.
# # 예를 들어, 정수 "3"은 데이터 내에서 세 번째로 빈번하게 사용된 단어를 나타냅니다. 
# # 이는 "가장 빈번하게 사용된 10,000 단어만을 고려하되 가장 많이 쓰인 20 단어는 제외"와 같은 빠른 필터링 작업을 가능케 합니다.
# # 관습에 따라 "0"은 특정 단어를 나타내는 것이 아니라 미확인 단어를 통칭합니다.

# # 2. word_size 전체데이터 부분 변경해서 최상값 확인
# # (x_train, y_train),(x_test,y_test) = imdb.load_data()
# # a = set()
# # x_train = pad_sequences(x_train, padding='pre')
# # x_test = pad_sequences(x_test, padding='pre')
# # a.update(np.unique(x_train))
# # a.update(np.unique(x_test))
# # print(len(a))
# # >>> print(len(a))
# # 88586

# # 3. 주간과제 : groupby()사용법 숙지
# # input data를 그룹별로 나누고 -> 집계함수 적용후 -> 적용데이터를 다시 합치는 함수
# # bbb = y_train_pd.groupby(0)[0].count()
# # y_train_pd의 0번째 column으로 같은 숫자끼리 묶은다음 이들의 0번째 column을(자기 자신) 사용하여 개수 count
# # 총 개수가 나온다.

# # 4. 인덱스를 단어로 바꿔주는 함수 찾을것
# # from keras.preprocessing.text import Tokenizer
# # token = Tokenizer()
# # token.fit_on_texts(imdb.get_word_index())
# # print(token.sequences_to_texts(x_train[0:1]))


# # 5. 125번, 126번 튠하기
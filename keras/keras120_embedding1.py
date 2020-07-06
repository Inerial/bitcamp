from keras.preprocessing.text import Tokenizer

text = "나는 맛있는 밥을 먹었다"

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)

x = token.texts_to_sequences([text])
print(x)

print(token.sequences_to_texts(x))

from keras.utils import to_categorical

word_size = len(token.word_index) + 1
x = to_categorical(x, num_classes=word_size)
print(x)

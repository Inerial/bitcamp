import tensorflow as tf
import numpy as np

# hihello
idx2char = ['e','h','i','l','o']

_data = np.array([['h','i','h','e','l','l','o']]).reshape(-1,1)

print(_data.shape)
print(_data)
print(type(_data))

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray()

print("==================")
print(_data)
print(_data.dtype)
print(type(_data))

x_data = _data[:6,]
y_data = _data[1:,]

print("============== x =============")
print(x_data)
print("============== y =============")
print(y_data)

y_data = np.argmax(y_data,axis=1)
print("============= y_argmax=========")
print(y_data)
print(y_data.shape)

x_data = x_data.reshape(1,6,5)
y_data = y_data.reshape(1,6)

print(x_data.shape)
print(y_data.shape)


seq_length = 6
input_dim = 5
output = 5
batch_size = 1
lr = 0.01
epoch = 1000

x = tf.compat.v1.placeholder(tf.float32, shape=[None,seq_length,input_dim])
y = tf.compat.v1.placeholder(tf.int32, shape=[None,seq_length])

##2. 모델 구성
# model.add(LSTM(output, input_shape=(6,5)))

# cell = tf.nn.rnn_cell.BasicLSTMCell(output)
cell = tf.keras.layers.LSTMCell(output)
hypothesis, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
print(hypothesis)
print(y)

weights = tf.ones([batch_size, seq_length])
seq_loss = tf.contrib.seq2seq.sequence_loss(
    logits = hypothesis, targets = y, weights = weights
)
cost = tf.reduce_mean(seq_loss)

train = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(cost)

prediction = tf.argmax(hypothesis, axis=2)

##3. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        ls, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})
        pred = sess.run(prediction, feed_dict={x:x_data})
        print('epoch :',i,', loss :', ls, ", prediction :", pred, ', true Y :', y_data)
    result_str = [idx2char[i] for i in np.squeeze(pred)]
    print('\nPrediction str :', ''.join(result_str))

# epoch : 399 , loss : 0.642205
# epoch : 400 , loss : 0.6421854
# [[2 1 0 3 3 4]]
# i
# h
# e
# l
# l
# o
# PS D:\Study>
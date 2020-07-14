import tensorflow as tf
import numpy as np

dataset = np.array([1,2,3,4,5,6,7,8,9,10])
print(dataset.shape)
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:i+size]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(dataset, 5)

x_data = dataset[:, :4].reshape(-1, 4, 1) 
y_data = dataset[:, 4:5] ## c랑은 다르게 대괄호 안에 ,로 구분한다.


# data = split_x(dataset, 5)
# print(data)

# x_data = np.array(data[:, :4, :])
# y_data = np.squeeze(data[:, 4:, :],axis=2)
print(x_data)
print(y_data)
input()


seq_length = 4
input_dim = 1
output = 100
batch_size = x_data.shape[0]
lr = 0.1
epoch = 1000

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4, 1])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
##2. 모델 구성
cell = tf.nn.rnn_cell.LSTMCell(output, state_is_tuple=True)
# cell = tf.keras.layers.LSTMCell(output, activation='elu')
# layer1 = tf.compat.v1.keras.layers.RNN(cell, return_state=True)(x)
# layer1= tf.keras.layers.RNN(cell)(x)
_,layer1= tf.nn.dynamic_rnn(cell,x, dtype=tf.float32)
# layer1 = layer1.call(inputs=x,initial_state=cell.get_initial_state(batch_size=batch_size, dtype=tf.float32))

print(layer1)
w2 = tf.Variable(tf.zeros([100, 100]), name='weight1')
b2 = tf.Variable(tf.zeros([100]), name='bias1')
layer2 = tf.nn.elu(tf.matmul(layer1,w2) + b2)

w3 = tf.Variable(tf.zeros([100, 100]), name='weight1')
b3 = tf.Variable(tf.zeros([100]), name='bias1')
layer3 = tf.nn.elu(tf.matmul(layer2,w3) + b3)

w4 = tf.Variable(tf.zeros([100, 100]), name='weight1')
b4 = tf.Variable(tf.zeros([100]), name='bias1')
layer4 = tf.nn.elu(tf.matmul(layer3,w4) + b4)

w5 = tf.Variable(tf.zeros([100, 1]), name='weight1')
b5 = tf.Variable(tf.zeros([1]), name='bias1')
hypothesis = tf.nn.elu(tf.matmul(layer4,w5) + b5)


print(hypothesis)
print(y)

# weights = tf.ones([batch_size, 4])

cost = tf.reduce_mean(tf.square(hypothesis - y))

train = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(cost)

# prediction = tf.argmax(hypothesis, axis=2)

##3. 훈련
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        ls, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})
        pred = sess.run(hypothesis, feed_dict={x:x_data})
        print('epoch :',i,', loss :', ls)
        print(pred)
        print(y_data)
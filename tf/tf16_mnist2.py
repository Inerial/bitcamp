import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
tf.set_random_seed(777)

(x_train, y_train),(x_test, y_tests) = mnist.load_data()

x_train = x_train.reshape(-1,28*28)/255
x_test = x_test.reshape(-1,28*28)/255

with tf.Session() as sess:
    y_train = sess.run(tf.one_hot(y_train,10))
    y_test = sess.run(tf.one_hot(y_tests,10))
y_train=y_train.reshape(-1,10)
y_test=y_test.reshape(-1,10)


x = tf.placeholder(tf.float32, shape = [None, 28*28])
y = tf.placeholder(tf.float32, shape = [None, 10])

w1 = tf.Variable(tf.truncated_normal([28*28, 2000], stddev=0.01), name='weight1')
b1 = tf.Variable(tf.constant(0.01,shape=[2000]), name='bias1')
layer1 = tf.nn.selu(tf.matmul(x,w1) + b1)
layer1 = tf.nn.dropout(layer1, keep_prob=0.2)

w2 = tf.Variable(tf.truncated_normal([2000, 2000], stddev=0.01), name='weight1')
b2 = tf.Variable(tf.constant(0.01, shape=[2000]), name='bias1')
layer2 = tf.nn.selu(tf.matmul(layer1,w2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob=0.2)

w3 = tf.Variable(tf.truncated_normal([2000, 2000], stddev=0.01), name='weight1')
b3 = tf.Variable(tf.constant(0.01, shape=[2000]), name='bias1')
layer3 = tf.nn.selu(tf.matmul(layer2,w3) + b3)
layer3 = tf.nn.dropout(layer3, keep_prob=0.2)

w4 = tf.Variable(tf.truncated_normal([2000, 1000], stddev=0.01), name='weight1')
b4 = tf.Variable(tf.constant(0.01, shape=[1000]), name='bias1')
layer4 = tf.nn.selu(tf.matmul(layer3,w4) + b4)
layer4 = tf.nn.dropout(layer4, keep_prob=0.2)

w5 = tf.Variable(tf.truncated_normal([1000, 1000], stddev=0.01), name='weight1')
b5 = tf.Variable(tf.constant(0.01, shape=[1000]), name='bias1')
layer5 = tf.nn.selu(tf.matmul(layer4,w5) + b5)
layer5 = tf.nn.dropout(layer5, keep_prob=0.2)

w6 = tf.Variable(tf.truncated_normal([1000, 1000], stddev=0.01), name='weight1')
b6 = tf.Variable(tf.constant(0.01, shape=[1000]), name='bias1')
layer6 = tf.nn.selu(tf.matmul(layer5,w6) + b6)
layer6 = tf.nn.dropout(layer6, keep_prob=0.2)

w7 = tf.Variable(tf.truncated_normal([1000, 1000], stddev=0.01), name='weight1')
b7 = tf.Variable(tf.constant(0.01, shape=[1000]), name='bias1')
layer7 = tf.nn.selu(tf.matmul(layer6,w7) + b7)
layer7 = tf.nn.dropout(layer7, keep_prob=0.2)

w8 = tf.Variable(tf.truncated_normal([1000, 500], stddev=0.01), name='weight1')
b8 = tf.Variable(tf.constant(0.01, shape=[500]), name='bias1')
layer8 = tf.nn.selu(tf.matmul(layer7,w8) + b8)
layer8 = tf.nn.dropout(layer8, keep_prob=0.2)

w9 = tf.Variable(tf.truncated_normal([500, 500], stddev=0.01), name='weight1')
b9 = tf.Variable(tf.constant(0.01, shape=[500]), name='bias1')
layer9 = tf.nn.selu(tf.matmul(layer8,w9) + b9)
layer9 = tf.nn.dropout(layer9, keep_prob=0.2)

w10 = tf.Variable(tf.truncated_normal([500, 500], stddev=0.01), name='weight1')
b10 = tf.Variable(tf.constant(0.01, shape=[500]), name='bias1')
layer10 = tf.nn.selu(tf.matmul(layer9,w10) + b10)
layer10 = tf.nn.dropout(layer10, keep_prob=0.2)

wf = tf.Variable(tf.truncated_normal([500, 10], stddev=0.001), name='weight2')
bf = tf.Variable(tf.constant(0.001, shape=[10]), name='bias2')
hypothesis = tf.nn.softmax(tf.matmul(layer10,wf) + bf)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1)) ## categorical_crossentropy

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(tf.argmax(hypothesis, axis=1), dtype = tf.float32)

batch_size = 500
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(40):
        for batch in range(int(60000/500)):
            cost_val, _ = sess.run([cost, train], feed_dict={x:x_train[batch*batch_size:(batch+1)*batch_size], y:y_train[batch*batch_size:(batch+1)*batch_size]})
            print(step, 'cost :', cost_val)#, '\n',hyp_val)

    hy, pred = sess.run([hypothesis ,predicted], feed_dict={x:x_test, y:y_test})
    # print(hy)
    print(pred)
    # print(y_test)
    print(accuracy_score(y_tests,pred))
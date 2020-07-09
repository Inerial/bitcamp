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

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
# (60000, 784) (10000, 784)
# (60000, 10) (10000, 10)

lr = 0.001
training_epochs = 40
batch_size = 500
total_batch = int(60000/batch_size)

x = tf.placeholder(tf.float32, shape = [None, 28*28])
y = tf.placeholder(tf.float32, shape = [None, 10])
keep_prob = tf.placeholder(tf.float32)

w1 = tf.get_variable("w1", shape=[784,2048],initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([2048]), tf.float32)
L1 = tf.nn.selu(tf.matmul(x,w1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

w2 = tf.get_variable("w2", shape=[2048,2048],initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([2048]), tf.float32)
L2 = tf.nn.selu(tf.matmul(L1,w2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

w3 = tf.get_variable("w3", shape=[2048,2048],initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([2048]), tf.float32)
L3 = tf.nn.selu(tf.matmul(L2,w3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

w4 = tf.get_variable("w4", shape=[2048,1024],initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([1024]), tf.float32)
L4 = tf.nn.selu(tf.matmul(L3,w4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

w5 = tf.get_variable("w5", shape=[1024,1024],initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([1024]), tf.float32)
L5 = tf.nn.selu(tf.matmul(L4,w5) + b5)
L5 = tf.nn.dropout(L5, keep_prob=keep_prob)

w6 = tf.get_variable("w6", shape=[1024,1024],initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([1024]), tf.float32)
L6 = tf.nn.selu(tf.matmul(L5,w6) + b6)
L6 = tf.nn.dropout(L6, keep_prob=keep_prob)

w7 = tf.get_variable("w7", shape=[1024,512],initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([512]), tf.float32)
L7 = tf.nn.selu(tf.matmul(L6,w7) + b7)
L7 = tf.nn.dropout(L7, keep_prob=keep_prob)

w8 = tf.get_variable("w8", shape=[512,512],initializer=tf.contrib.layers.xavier_initializer())
b8 = tf.Variable(tf.random_normal([512]), tf.float32)
L8 = tf.nn.selu(tf.matmul(L7,w8) + b8)
L8 = tf.nn.dropout(L8, keep_prob=keep_prob)

w9 = tf.get_variable("w9", shape=[512,512],initializer=tf.contrib.layers.xavier_initializer())
b9 = tf.Variable(tf.random_normal([512]), tf.float32)
L9 = tf.nn.selu(tf.matmul(L8,w9) + b9)
L9 = tf.nn.dropout(L9, keep_prob=keep_prob)

w10 = tf.get_variable("w10", shape=[512,10],initializer=tf.contrib.layers.xavier_initializer())
b10 = tf.Variable(tf.random_normal([10]), tf.float32)
hypothesis = tf.nn.softmax(tf.matmul(L9,w10) + b10)


cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1)) ## categorical_crossentropy

train = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

predicted = tf.cast(tf.argmax(hypothesis, axis=1), dtype = tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(training_epochs):
        avg_cost = 0
        for batch in range(total_batch):
            batch_xs, batch_ys = x_train[batch*batch_size : (batch+1)*batch_size], y_train[batch*batch_size : (batch+1)*batch_size]
            feed_dict = {x:batch_xs, y:batch_ys, keep_prob:0.8}
            cost_val, _ = sess.run([cost, train], feed_dict=feed_dict)
            print(step, 'cost :', cost_val)#, '\n',hyp_val)
            avg_cost += cost_val/total_batch
        print('Epo :', '%04d' %(step+1), 'cost :', '{:.9f}'.format(avg_cost))
    hy, pred = sess.run([hypothesis ,predicted], feed_dict={x:x_test, y:y_test, keep_prob:0.8})
    # print(hy)
    print(pred)
    # print(y_test)
    print(accuracy_score(y_tests,pred))
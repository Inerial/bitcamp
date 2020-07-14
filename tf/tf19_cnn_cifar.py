## 주말과제 cifar10 구현

import tensorflow as tf
import numpy as np
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
tf.set_random_seed(777)

(x_train, y_train),(x_test, y_tests) = cifar10.load_data()

x_train = x_train/255
x_test = x_test/255

with tf.Session() as sess:
    y_train = sess.run(tf.one_hot(y_train,10))
    y_test = sess.run(tf.one_hot(y_tests,10))
y_train=y_train.reshape(-1,10)
y_test=y_test.reshape(-1,10)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

lr = 0.00015
training_epochs = 50
batch_size = 500
total_batch = int(50000/batch_size)

x_imag = tf.placeholder(tf.float32, shape = [None, 32,32,3])
# x_imag = tf.reshape(x, [-1,32,32,3])
y = tf.placeholder(tf.float32, shape = [None, 10])
keep_prob = tf.placeholder(tf.float32)

w1 = tf.get_variable("w1", shape=[3,3,3,32],initializer=tf.contrib.layers.xavier_initializer())
L1 = tf.nn.conv2d(x_imag, w1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.elu(L1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

w2 = tf.get_variable("w2", shape=[3,3,32,32],initializer=tf.contrib.layers.xavier_initializer())
L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.elu(L2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
L2 = tf.nn.max_pool2d(L2,(1,2,2,1), strides = [1,2,2,1], padding='VALID')


w3 = tf.get_variable("w3", shape=[3,3,32,64],initializer=tf.contrib.layers.xavier_initializer())
L3 = tf.nn.conv2d(L2, w3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.elu(L3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

w4 = tf.get_variable("w4", shape=[3,3,64,64],initializer=tf.contrib.layers.xavier_initializer())
L4 = tf.nn.conv2d(L3, w4, strides=[1,1,1,1], padding='SAME')
L4 = tf.nn.elu(L4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
L4 = tf.nn.max_pool2d(L4,(1,2,2,1), strides = [1,2,2,1], padding='VALID')


w5 = tf.get_variable("w5", shape=[3,3,64,128],initializer=tf.contrib.layers.xavier_initializer())
L5 = tf.nn.conv2d(L4, w5, strides=[1,1,1,1], padding='SAME')
L5 = tf.nn.elu(L5)
L5 = tf.nn.dropout(L5, keep_prob=keep_prob)

w6 = tf.get_variable("w6", shape=[3,3,128,128],initializer=tf.contrib.layers.xavier_initializer())
L6 = tf.nn.conv2d(L5, w6, strides=[1,1,1,1], padding='SAME')
L6 = tf.nn.elu(L6)
L6 = tf.nn.dropout(L6, keep_prob=keep_prob)
L6 = tf.nn.max_pool2d(L6,(1,2,2,1), strides = [1,2,2,1], padding='VALID')


w7 = tf.get_variable("w7", shape=[3,3,128,256],initializer=tf.contrib.layers.xavier_initializer())
L7 = tf.nn.conv2d(L6, w7, strides=[1,1,1,1], padding='SAME')
L7 = tf.nn.elu(L7)
L7 = tf.nn.dropout(L7, keep_prob=keep_prob)

w8 = tf.get_variable("w8", shape=[3,3,256,256],initializer=tf.contrib.layers.xavier_initializer())
L8 = tf.nn.conv2d(L7, w8, strides=[1,1,1,1], padding='SAME')
L8 = tf.nn.elu(L8)
L8 = tf.nn.dropout(L8, keep_prob=keep_prob)


print(L8)
L8= tf.reshape(L8, [-1, 4*4*256])

w9 = tf.get_variable("w9", shape=[4*4*256,256],initializer=tf.contrib.layers.xavier_initializer())
b9 = tf.Variable(tf.random_normal([256]), tf.float32)
L9 = tf.nn.elu(tf.matmul(L8,w9) + b9)
L9 = tf.nn.dropout(L9, keep_prob=keep_prob)

w10 = tf.get_variable("w10", shape=[256,32],initializer=tf.contrib.layers.xavier_initializer())
b10 = tf.Variable(tf.random_normal([32]), tf.float32)
L10 = tf.nn.elu(tf.matmul(L9,w10) + b10)
L10 = tf.nn.dropout(L10, keep_prob=keep_prob)

w11 = tf.get_variable("w11", shape=[32,10],initializer=tf.contrib.layers.xavier_initializer())
b11 = tf.Variable(tf.random_normal([10]), tf.float32)
hypothesis = tf.nn.softmax(tf.matmul(L10,w11) + b11)


cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1)) ## categorical_crossentropy

train = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

predicted = tf.cast(tf.argmax(hypothesis, axis=1), dtype = tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(training_epochs):
        avg_cost = 0
        for batch in range(total_batch):
            batch_xs, batch_ys = x_train[batch*batch_size : (batch+1)*batch_size], y_train[batch*batch_size : (batch+1)*batch_size]
            feed_dict = {x_imag:batch_xs, y:batch_ys, keep_prob:0.8}
            cost_val, _ = sess.run([cost, train], feed_dict=feed_dict)
            print(step, 'cost :', cost_val)#, '\n',hyp_val)
            avg_cost += cost_val/total_batch
        print('Epo :', '%04d' %(step+1), 'cost :', '{:.9f}'.format(avg_cost))
    hy, pred = sess.run([hypothesis ,predicted], feed_dict={x_imag:x_test, y:y_test, keep_prob:0.8})
    # print(hy)
    print(pred)
    # print(y_test)
    print(accuracy_score(y_tests,pred))

    ## keras에서 평가시에는 dropout이 적용되지 않는다.

# 49 cost : 0.44937414
# 49 cost : 0.39924458
# 49 cost : 0.43283626
# 49 cost : 0.3728475
# 49 cost : 0.48546007
# 49 cost : 0.42134932
# Epo : 0050 cost : 0.463952407
# [5. 8. 8. ... 5. 4. 7.]
# 0.7519
# PS D:\Study>
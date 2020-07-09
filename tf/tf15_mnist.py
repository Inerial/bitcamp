import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
tf.set_random_seed(777)

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1,28*28)/255
x_test = x_test.reshape(-1,28*28)/255

tf.
x = tf.placeholder(tf.float32, shape = [None, 28*28])
y = tf.placeholder(tf.float32, shape = [None, 10])

w1 = tf.Variable(tf.zeros([28*28, 2000]), name='weight1')
b1 = tf.Variable(tf.zeros([2000]), name='bias1')
layer1 = tf.nn.relu(tf.matmul(x,w1) + b1)

w2 = tf.Variable(tf.zeros([2000, 2000]), name='weight1')
b2 = tf.Variable(tf.zeros([2000]), name='bias1')
layer2 = tf.nn.relu(tf.matmul(layer1,w2) + b2)

w3 = tf.Variable(tf.zeros([2000, 2000]), name='weight1')
b3 = tf.Variable(tf.zeros([2000]), name='bias1')
layer3 = tf.nn.relu(tf.matmul(layer2,w3) + b3)

w4 = tf.Variable(tf.zeros([2000, 1000]), name='weight1')
b4 = tf.Variable(tf.zeros([1000]), name='bias1')
layer4 = tf.nn.relu(tf.matmul(layer3,w4) + b4)

w5 = tf.Variable(tf.zeros([1000, 1000]), name='weight1')
b5 = tf.Variable(tf.zeros([1000]), name='bias1')
layer5 = tf.nn.relu(tf.matmul(layer4,w5) + b5)

w6 = tf.Variable(tf.zeros([1000, 1000]), name='weight1')
b6 = tf.Variable(tf.zeros([1000]), name='bias1')
layer6 = tf.nn.relu(tf.matmul(layer5,w6) + b6)

w7 = tf.Variable(tf.zeros([1000, 1000]), name='weight1')
b7 = tf.Variable(tf.zeros([1000]), name='bias1')
layer7 = tf.nn.relu(tf.matmul(layer6,w7) + b7)

w8 = tf.Variable(tf.zeros([1000, 500]), name='weight1')
b8 = tf.Variable(tf.zeros([500]), name='bias1')
layer8 = tf.nn.relu(tf.matmul(layer7,w8) + b8)

w9 = tf.Variable(tf.zeros([500, 500]), name='weight1')
b9 = tf.Variable(tf.zeros([500]), name='bias1')
layer9 = tf.nn.relu(tf.matmul(layer8,w9) + b9)

w10 = tf.Variable(tf.zeros([500, 500]), name='weight1')
b10 = tf.Variable(tf.zeros([500]), name='bias1')
layer10 = tf.nn.relu(tf.matmul(layer9,w10) + b10)

wf = tf.Variable(tf.zeros([500, 10]), name='weight2')
bf = tf.Variable(tf.zeros([10]), name='bias2')
hypothesis = tf.nn.softmax(tf.matmul(layer10,wf) + bf)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1)) ## categorical_crossentropy

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(tf.argmax(hypothesis, axis=1), dtype = tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_train = sess.run(tf.one_hot(y_train, 10))
    y_tests = sess.run(tf.one_hot(y_test, 10))
    for step in range(10):
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_train, y:y_train})
        print(step, 'cost :', cost_val)#, '\n',hyp_val)

    hy, pred = sess.run([hypothesis ,predicted], feed_dict={x:x_test, y:y_tests})
    # print(hy)
    # print(pred)
    # print(y_test)
    print(accuracy_score(y_test,pred))
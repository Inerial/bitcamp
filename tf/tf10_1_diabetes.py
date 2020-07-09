from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import tensorflow as tf
import numpy as np

diabetes = load_diabetes()

x_data = diabetes.data
y_data = diabetes.target.reshape(-1,1)
print(x_data.shape)
print(y_data.shape)

x = tf.placeholder(tf.float32, shape = [None, 10])
y = tf.placeholder(tf.float32, shape = [None, ])


w = tf.Variable(tf.random_normal([10, ]), name='weight1')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x,w) + b


cost = tf.reduce_mean(tf.square(hypothesis - y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(5000):
        _, cost_val, hyp_val = sess.run([train, cost, hypothesis], feed_dict={x:x_data, y:y_data})
        if step % 20 == 0:
            print(step, 'cost :', cost_val)#, '\n',hyp_val)

    pred = sess.run(hypothesis, feed_dict={x:x_data})

    print('score :',r2_score(y_data,pred))
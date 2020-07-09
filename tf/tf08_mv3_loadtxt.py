import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

dataset = np.loadtxt('./data/csv/data-01-test-score.csv', delimiter=',', dtype=np.float32)

x_data = dataset[:,0:-1]

y_data = dataset[:,[-1]]

x = tf.placeholder(tf.float32, shape=[None, x_data.shape[1]])
y = tf.placeholder(tf.float32, shape=[None, y_data.shape[1]])

w = tf.Variable(tf.random_normal([x_data.shape[1], 1]), name='weight1')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x,w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.00005).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10000):
        _, cost_val, hyp_val = sess.run([train, cost, hypothesis], feed_dict={x:x_data, y:y_data})
        if step % 20 == 0:
            print(step, 'cost :', cost_val)#, '\n',hyp_val)
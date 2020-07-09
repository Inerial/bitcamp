import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
tf.set_random_seed(777)
iris = load_iris()

x_data = iris.data
y_data = iris.target

x_train,x_test, y_train,y_test = train_test_split(
    x_data,y_data,shuffle=True, train_size=0.8
)

x = tf.placeholder(tf.float32, shape=[None,4])
y = tf.placeholder(tf.float32, shape=[None,3])

w = tf.Variable(tf.random_normal([4, 3]), name='weight1')
b = tf.Variable(tf.random_normal([3]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1)) ## categorical_crossentropy

train = tf.train.GradientDescentOptimizer(learning_rate=0.2).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_train = sess.run(tf.one_hot(y_train, 3))
    y_test = sess.run(tf.one_hot(y_test, 3))
    for step in range(10000):
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_train, y:y_train})
        if step % 20 == 0:
            print(step, 'cost :', cost_val)#, '\n',hyp_val)

    hy, pred = sess.run([hypothesis ,predicted], feed_dict={x:x_test, y:y_test})
    print(hy)
    print(pred)
    print(accuracy_score(y_test,pred))



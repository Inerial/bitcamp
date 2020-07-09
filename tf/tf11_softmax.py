import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]

y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]

x = tf.placeholder(tf.float32, shape=[None,4])
y = tf.placeholder(tf.float32, shape=[None,3])

w = tf.Variable(tf.random_normal([4, 3]), name='weight1')
b = tf.Variable(tf.random_normal([3]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1)) ## categorical_crossentropy

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10000):
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})
        if step % 20 == 0:
            print(step, 'cost :', cost_val)#, '\n',hyp_val)

    hy, pred, acc = sess.run([hypothesis ,predicted, accuracy], feed_dict={x:x_data, y:y_data})
    print(hy)
    print(pred)
    print(acc)

    # a = sess.run(hypothesis, feed_dict={x:[[1,11,7,9]]})
    # print(a , sess.run(tf.argmax(a,1)))

    # b = sess.run(hypothesis, feed_dict={x:[[1,3,4,3]]})
    # print(b , sess.run(tf.argmax(b,1)))

    # c = sess.run(hypothesis, feed_dict={x:[[11,33,4,13]]})
    # print(c , sess.run(tf.argmax(c,1)))

    
    alls = sess.run(hypothesis, feed_dict={x:[[1,11,7,9],[1,3,4,3],[11,33,4,13]]})
    print(alls , sess.run(tf.argmax(alls,1)))
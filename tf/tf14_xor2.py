import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype = np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)


x = tf.placeholder(tf.float32, shape = [None, 2])
y = tf.placeholder(tf.float32, shape = [None, 1])

w1 = tf.Variable(tf.random_normal([2, 100]), name='weight1')
b1 = tf.Variable(tf.random_normal([100]), name='bias1')
layer1 = tf.matmul(x,w1) + b1

w2 = tf.Variable(tf.random_normal([100, 50]), name='weight1')
b2 = tf.Variable(tf.random_normal([50]), name='bias1')
layer2 = tf.matmul(layer1,w2) + b2

w3 = tf.Variable(tf.zeros([50, 1]), name='weight2')
b3 = tf.Variable(tf.zeros([1]), name='bias2')
hypothesis = tf.sigmoid(tf.matmul(layer2,w3) + b3)

cost = - tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis)) ## binary_crossentropy

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(5000):
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})
        if step % 20 == 0:
            print(step, 'cost :', cost_val)#, '\n',hyp_val)

    hy, pred, acc = sess.run([hypothesis ,predicted, accuracy], feed_dict={x:x_data, y:y_data})
    print(hy)
    print(pred)
    print(acc)
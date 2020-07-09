from sklearn.datasets import load_breast_cancer
import tensorflow as tf

breast = load_breast_cancer()

x_data = breast.data
y_data = breast.target.reshape(-1,1)
print(x_data.shape)
print(y_data.shape)

x = tf.placeholder(tf.float32, shape = [None, 30])
y = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.zeros([30, 1]), name='weight1')
b = tf.Variable(tf.zeros([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x,w) + b)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(hypothesis, feed_dict={x:x_data}))

cost = - tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis)) ## binary_crossentropy

train = tf.train.GradientDescentOptimizer(learning_rate=0.0000001).minimize(cost)

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
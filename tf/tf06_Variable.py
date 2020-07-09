import tensorflow as tf
tf.set_random_seed(777)
x_train = [1,2,3]
y_train = [3,5,7]
x_test = [4,5,6]
x_test = [9,11,13]

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

w = tf.Variable([0.3], tf.float32)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(w))

sess.close()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = w.eval()
print(bbb)
sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = w.eval(session=sess)
print(ccc)
sess.close()
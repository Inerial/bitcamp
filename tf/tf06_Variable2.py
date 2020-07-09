import tensorflow.compat.v1 as tf
tf.set_random_seed(777)

x = [1.,2.,3.]
w = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.], tf.float32)

hypothesis = x * w + b

sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(hypothesis)
print(aaa)
sess.close()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = hypothesis.eval()
print(bbb)
sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = hypothesis.eval(session=sess)
print(ccc)
sess.close()
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

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(w))

hypothesis = x * w + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

lr = 0.175
train = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)
## 해당 옵티마이저의 코스트를 최소로 만드는것

with tf.Session() as sess: ## with 안에서만 쓰는거, enter와 exit를 해야되는 함수에서 exit를 무조건 실행시켜줌
    sess.run(tf.global_variables_initializer())
    sess.run(tf.global_variables_initializer())

    for step in range(300):
        _, cost_val, w_val, b_val = sess.run([train, cost, w, b], feed_dict={x:x_train, y:y_train})
        # sess.run(feed)
        if step % 20 == 0:
            print(step, cost_val, w_val, b_val)
    print(sess.run(hypothesis, feed_dict={x:[4]}))
    print(sess.run(hypothesis, feed_dict={x:[5,6]}))
    print(sess.run(hypothesis, feed_dict={x:[6,7,8]}))

## predict
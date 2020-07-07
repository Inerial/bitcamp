import tensorflow as tf

node0 = tf.constant(2.0, tf.float32)
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0, tf.float32)
node3 = tf.constant(5.0, tf.float32)

node4 = tf.add_n([node1,node2,node3])
node5 = tf.subtract(node2, node1)
node6 = tf.multiply(node1, node2)
node7 = tf.divide(node2, node0)

print("node1 :", node1, "node2 :", node2)
print("node3 : ", node3)

sess = tf.Session()
print("sess.run(node0,node1,node2,node3)  :", sess.run([node0,node1,node2,node3]))
print("sess.run(node4) :", sess.run(node4))
print("sess.run(node5) :", sess.run(node5))
print("sess.run(node6) :", sess.run(node6))
print("sess.run(node7) :", sess.run(node7))
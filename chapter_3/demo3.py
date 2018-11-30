import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")

with tf.Session() as sess:

    result = a + b
    sess.run(result)
    print(result.eval())

# sess = tf.Session()
# sess.run(result)
# print(result)
# print(sess)
# sess.close()

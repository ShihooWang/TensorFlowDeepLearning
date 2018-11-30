
# 在神经网络的优化算法中，最常使用反向传播算法
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1))

# 定义placeholder作为存放输入数据的地方，这里的维度也不一定要定义
# 但如果维度是确定的，那么给出维度可以降低出错的概率
x = tf.placeholder(tf.float32, shape=(1, 2), name="input")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
# print(sess.run(y))

result = sess.run(y, feed_dict={x: [[0.7, 0.9]]})
print(result)
sess.close()

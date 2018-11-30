# 解决二分类问题
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from numpy.random import RandomState

# 定义训练数据batch的大小
batch_size = 8

# 定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 在shape的一个维度上使用None可以反驳使用不大的batch大小。在训练时需要吧数据分成比较小的batch，
# 但是在测试时，可以一次性使用全部的数据。当数据集比较小时，这样比较方便测试，但数据集比较大时，
# 将大量数据放入一个batch可能会导致内测溢出。
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y_input')

# 定义神经网络向前传播的过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播的算法
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

# 定义规则来给出样本的标签
# 在这里所有x1+x2<1的样例都被认为是正样本（比如零件合格），而其他为负样本（比如零件不合格）。
# 在这里使用0来表示负样本，1来表示正样本。大部分解决分类问题的神经网络都会采用0和1的表示方法
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

# 创建一个会话来运行TensorFlow程序
with tf.Session() as sess:
    # 初始化变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))

    # 设定训练轮数
    STEPS = 5000
    for i in range(STEPS):
        # 每次选取batch_size个样本
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step,
                 feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(
                cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training steps, cross entropy on all data is %g",(i, total_cross_entropy))

    print(sess.run(w1))
    print(sess.run(w2))

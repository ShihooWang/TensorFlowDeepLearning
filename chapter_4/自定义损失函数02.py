import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 程序实现了一个拥有两个输入节点、一个输出节点，没有隐藏层的神经网络
import tensorflow as tf
from numpy.random import RandomState

batch_size = 8
# 两个输入节点
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
# 回归问题一般只有一个输出节点 y_ 为标准答案
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

# 使用变量声明函数Variable 给出了初始化这个变量的方法
# 产生一个2 X 1的矩阵，矩阵中元素是均值为0，标准差为2，随机种子为1
# 定义了一个单层的神经网络向前传播过程，这里就是简单加权和
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 定义预测多了和预测少了的成本
loss_less = 10
loss_more = 1
# 定义了损失函数 来刻画预测值和真实值的差距
loss = tf.reduce_mean(tf.where(tf.greater(y, y_), (y - y_)*loss_more, (y_ - y)*loss_less))
# 反向传播算法 来优化神经网络中的参数 学习率为0.001
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 设置回归的正确值为两个输入的和加上一个随机量。之所以要加上一个随机量，是为了加入不可预测的噪音
# 否则不同损失函数的意义就不大了，因为不同损失函数都会在能完全预测正确的时候最低。
# 一般来说噪音为一个均值为0的小量，所以这里的噪音设置为 -0.05-0.05的随机数
Y = [[x1 + x2 + rdm.rand()/10.0 - 0.05] for (x1, x2) in X]

# 训练神经网络
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 5000
    for i in range(STEPS):
        # 每次选取batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)
        # 通过选取的样本训练神经网络并更新参数 feed_dict来指定取值。
        # feed_dict 是一个字典（map），在字典中，需要给出每隔用到的placeholder的取值
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
    print(sess.run(w1))

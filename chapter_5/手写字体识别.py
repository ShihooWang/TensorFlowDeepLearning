import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST 数据集相关的常数
INPUT_NODE = 784        # 输入层的节点数，相当于MNIST图片的像素28*28
OUTPUT_NODE = 10        # 输出层的节点数，对应0-9这10个数

# 配置神经网络参数
LAYER1_NODE = 500       # 隐藏层节点数，这里只使用一个隐藏层的网络结构作为样例 隐藏层有500个节点

BATCH_SIZE = 100        # 一个训练batch的大小。数字越小，训练过程越接近随机梯度下降，越大，越接近梯度下降

# LEARNING_RATE_BASE = 0.8        # 基础学习率
# LEARNING_RATE_DECAY = 0.99        # 基础学习率的衰减率
# REGULARIZATION_RATE = 0.0001        # 描述模型复杂程度的正则化项在损失函数中的系数
# TRAINING_STEPS = 30000              # 训练轮数
# MOVING_AVERAGE_DECAY = 0.99         # 滑动平均衰减率

# 一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的向前传播结果。
# 这里定义了一个使用ReLU激活函数的三层全连接神经网络。通过加入隐藏层时间了多层网络结构，通过ReLU函数实现了去线性化。
# 在这个函数中也支持传入用于计算参数平均值的类，这样方便在测试的时候使用滑动平均模型。
def inference(input_tensor, avg_class, weight1, biases1, weight2, biases2):
    # 当没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        # 计算隐藏层的向前传播结果,这里使用了ReLU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weight1) + biases1)

        # 计算输出层的向前传播结果，这里没有加入激活函数和softmax分类器
        return tf.matmul(layer1, weight2) + biases2
    else:
        # 首先使用avg_class.average函数来计算得出变量的滑动平均值，然后再计算向前传播的结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weight1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weight2)) + avg_class.average(biases2)

# 训练模型的过程
def train(mnist) :
    y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_NODE], name="y-input")
    x = tf.placeholder(tf.float32, shape=[None, INPUT_NODE], name="x-input")

    # 生成隐藏层的参数
    weight1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 生成输出层的参数
    weight2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.01))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算在当前参数下神经网络向前传播的结果，这里用于计算滑动平均的类为None
    # 所以函数不会使用参数的滑动平均值
    y = inference(x, None, weight1, biases1, weight2, biases2)

    # 定义村塾训练轮数的变量，这个变量不需要计算滑动平均值。在tf中，一般代表训练轮数的变量设置为不接训练的参数
    global_step = tf.Variable(0, trainable=False)
    # 初始化滑动平均类
    # variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages = tf.train.ExponentialMovingAverage(0.99, global_step)
    # 在所有代表神经网络参数的变量上使用滑动平均
    variables_average_op = variable_averages.apply(tf.trainable_variables())
    # 计算使用了滑动平均之后的向前传播结果。
    average_y = inference(x, variable_averages, weight1, biases1, weight2, biases2)

    # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 计算在当前batch中所有样例的交叉熵的平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    # regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularizer = tf.contrib.layers.l2_regularizer(0.0001)
    # 计算模型的正则化损失 计算神经网络权重上的正则化损失，而不使用偏置项
    regularization = regularizer(weight1) + regularizer(weight2)
    # 总损失等于交叉熵损失和正则化损失和
    loss = cross_entropy_mean + regularization
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        # LEARNING_RATE_BASE,
        0.8,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,  # 需要学习多少轮
        0.99
        # LEARNING_RATE_DECAY
    )
    # 优化损失函数 梯度下降
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 在训练神经网络模型时，每过一遍数据既要通过反向传播来更新神经网络中的参数，又要更新没有个参数的滑动平均值。
    with tf.control_dependencies([train_step, variables_average_op]):
        train_op = tf.no_op(name="train")

    # 检验使用了滑动平均模型的神经网络前向传播结果是否正确
    # 计算每一个样本的预测答案
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # 正确率 使用cast转换为float类型
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 准备验证数据
        validete_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        # 准备测试数据
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 迭代训练神经网络
        # for i in range(TRAINING_STEPS):
        for i in range(30000):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validete_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))

            # 产生这一轮使用一个batch的训练数据，并运行训练过程
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 在训练结束之后，在测试数据上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        # print("After %d training step(s), validation accuracy using average model is %g " % (TRAINING_STEPS, test_acc))
        print("After %d training step(s), validation accuracy using average model is %g " % (30000, test_acc))

# 主程序入口
def main(argv = None):
    mnist = input_data.read_data_sets("C:/Wang_File/MNIST", one_hot=True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()





























import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
使用LeNet-5模型实现卷积神经网络的前向传播
"""
import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5

# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5

# 全连接层的节点个数
FC_SIZE = 512

# 定义卷积神经网络的前向传播过程。 注意区别训练过程和测试过程。
def infernece(input_tensor, train, regularizer):

    # 声明第一层卷积层的变量，并实现前向传播过程。
    # 定义的卷积层输入为 28*28*1 的原始MNIST图片像素， 使用了全0填充，所以输出为 28*28*32
    with tf.variable_scope("layer1-conv1"):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable(
            "bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

        # 使用边长为5 深度为32的过滤器，移动步长1 且使用全填充
        conv1 = tf.nn.conv2d(
            input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 实现第二层池化层的前向传播，使用了最大池化层。过滤器边长为2，使用全0填充，步长为2
    # 这一层的输入为上一层的输出 即 28*28*32 ，输出这为 14*14*32 池化不会改变深度
    with tf.name_scope("layer2-pooll"):
        pool1 = tf.nn.max_pool(
            relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 声明第三层卷积层的变量，并实现前向传播。
    # 这一层的输入为  14*14*32  输出为  14*14*64
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable(
            "bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

        # 使用边长为5， 深度为64的过滤器，步长为1，使用全0填充
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 实现第四层池化层的前向传播过程。和第二层类似，这层的输入为 14*14*64 输出为 7*7*64
    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(
            relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 将第四层池化层的输出转化为第五层全连接层的输入格式。第四层输出 7*7*64
    # 这里需要将 7*7*64 拉成一个向量。
    pool_shape = pool2.get_shape().as_list()

    # 拉成一个向量，长度为之前矩阵的长宽及深度的乘积 pool_shape[0]为一个batch中数据个数  这里可以理解吧
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    # 通过tf.reshape 函数将第四层的输出变成一个batch的向量
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # 声明第五层全连接层的变量，并实现前向传播过程。输入的长度为7*7*64 = 3136 输出是512长度
    with tf.variable_scope("layer5-fc1"):
        fc1_weights = tf.get_variable(
            "weight", [nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc1_weights))
        fc1_biases = tf.get_variable(
            "bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

        # 使用dropout在训练时随机将部分节点的输出改为0，避免过拟合。一般用在全连接层
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    # 声明第六层全连接的变量和前向传播过程。这层长度为512，输出为长度10.通过Softmax分类得到结果
    with tf.variable_scope("layer6-fc2"):
        fc2_weights = tf.get_variable(
            "weight", [FC_SIZE, NUM_LABELS], initializer=tf.truncated_normal_initializer(0.1))
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc2_weights))
        fc2_biases = tf.get_variable(
            "bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        # 这里不需要激活函数
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit
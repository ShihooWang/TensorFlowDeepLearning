# 该类用来定义前向传播的过程及神经网络中的参数
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

# 使用该函数来获取变量
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        "weights", shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1))

    # 将变量的正则化损失加入损失集合
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))

    return weights

# 定义神经网络的前向传播过程
def inference(input_tensor, regularizer):

    # 声明第一层神经网络的变量并完成前向传播过程
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    # 声明第二层
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2
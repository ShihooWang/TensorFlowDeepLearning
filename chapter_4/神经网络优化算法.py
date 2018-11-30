import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 反向传播算法(backpropagation) 梯度下降(gradient decent)
# 调整神经网络中参数的取值

# 示例 通过集合计算一个5层神经网络带L2正则化的损失函数的计算方法
import tensorflow as tf


# 获取一层神经网络边上的权重，并将这个权重的L2正则化损失加入名称为‘losses’的集合中
def get_weight(shape, w):
    # 生成一个变量
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    # add_to_collection 函数将这个新生成的变量L2正则化损失项，加入集合
    # 该函数第一个参数‘losses’是集合的名称，第二个参数是要加入该集合的内容
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(w)(var))
    return var


# 两个输入点
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
batch_size = 8

# 定义了每一层网络中节点的个数
layer_dimension = [2, 10, 10, 10, 1]
# 神经网络的层数
n_layers = len(layer_dimension)
# 这个变量维护向前出传播时最深层的节点，开始的时候就是输入层
cur_layer = x
# 当前层的节点个数
in_dimension = layer_dimension[0]

# 通过一个for循环来生成5个全连接的神经网络结构
for i in range(1, n_layers):
    # layer_dimension[i] 为下一层的节点个数
    out_dimension = layer_dimension[i]
    # 生成当前层中权重变量，并将这个变量的L2正则化损失加入计算图上的集合
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    # 使用ReLU激活函数
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    # 进入下一层之前将下一层的节点个数更新为当前层节点个数
    # in_dimension = layer_dimension[i]
    in_dimension = out_dimension

# 在定义神经网络前向传播的同时，已经将所有的 L2正则化损失加入了图上的集合
# 这里只需要计算刻画模型在训练数据上表现的损失函数
mse_less = tf.reduce_mean(tf.square(y_ - cur_layer))
# 将均方误差损失函数加入损失集合
tf.add_to_collection('losses', mse_less)
# get_collection 返回一个列表，这个列表是所有这个集合中的元素。
# 在这个样例中，这些元素就是损失函数的不同部分，将它们加起来就可以得到最终的损失函数。
loss = tf.add_n(tf.get_collection('losses'))
print(loss)

# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     result = sess.run(loss)
#     print(result)





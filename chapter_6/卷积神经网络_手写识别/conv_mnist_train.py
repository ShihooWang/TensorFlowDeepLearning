import tensorflow as tf
from chapter_6.卷积神经网络_手写识别 import conv_mnist_infernece
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 配置神经网络参数
BATCH_SIZE  = 100  # 数据模型
LEARNINT_RATE_BASE = 0.8  # 学习率
LEARNINT_RATE_DECAY = 0.99  # 学习衰减率
REGULARAZTION_RATE = 0.0001  # 正则化损失
TRAINING_STEPS = 30000  # 训练步数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率

# 保存模型路径
MODEL_SAVE_PATH = "C:/Wang_File/conv/"
MODEL_NAME = "model.ckpt"

def train(mnist):
    # 调整输入数据的格式，输入为一个四维矩阵
    x = tf.placeholder(dtype=tf.float32,
                       shape=[BATCH_SIZE,
                              conv_mnist_infernece.IMAGE_SIZE,
                              conv_mnist_infernece.IMAGE_SIZE,
                              conv_mnist_infernece.NUM_CHANNELS],
                       name="x-input")
    y_ = tf.placeholder(tf.float32, [None, conv_mnist_infernece.OUTPUT_NODE], name='y-out')

    # 计算L2正则化损失项
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = conv_mnist_infernece.infernece(x, False, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作 及训练过程
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 指数衰减
    learning_rate = tf.train.exponential_decay(
        LEARNINT_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNINT_RATE_DECAY)
    # 在训练神经网络模型时，每过一遍数据既要通过反向传播来更新神经网络中的参数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 同时也要更新每一个参数的滑动平均值。tf.group 和 tf.control_dependencies 两种方法
    # train_op = tf.group(train_step, variavle_averages_op, name='train')
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 开始训练过程
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # print(xs.shape)
            # print(ys.shape)
            reshape_xs = tf.reshape(xs, [BATCH_SIZE,
                                         conv_mnist_infernece.IMAGE_SIZE,
                                         conv_mnist_infernece.IMAGE_SIZE,
                                         conv_mnist_infernece.NUM_CHANNELS])
            reshape_xs = reshape_xs.eval()
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshape_xs, y_: ys})
            if i % 10 == 0:
                print("After %d training step(s), loss on training average batch is %g ." % (step, loss_value))
                # 保存模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("C:/Wang_File/MNIST", one_hot=True)
    train(mnist)


if __name__ == "__main__":
    tf.app.run()
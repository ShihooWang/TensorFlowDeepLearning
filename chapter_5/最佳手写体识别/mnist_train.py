import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
# import mnist_inference
# python console中配置信息 sys.path.extend(['C:\\Wang_Work\\Pycharm_workSpace\\Demo1', 'C:/Wang_Work/Pycharm_workSpace/Demo1'])
from chapter_5.最佳手写体识别 import mnist_inference


# 配置神经网络参数
BATCH_SIZE  = 100
LEARNINT_RATE_BASE = 0.8
LEARNINT_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
# 保存模型路径
MODEL_SAVE_PATH = "C:/Wang_File/model/"
MODEL_NAME = "model.ckpt"

# 定义训练过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NONE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NONE], name='y-out')

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作 及训练过程
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())  # 所有可训练的变量
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)  # 交叉熵的均值
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

    # 初始化tf 持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 开始训练过程
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training average batch is %g ." % (step, loss_value))
                # 保存模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets("C:/Wang_File/MNIST", one_hot=True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()

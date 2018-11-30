import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
from chapter_9 import mnist_inference

BATCH_SIZE  = 100
LEARNINT_RATE_BASE = 0.8
LEARNINT_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 3000
# TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
# 保存模型路径
MODEL_SAVE_PATH = "C:/Wang_File/model/"
MODEL_NAME = "model.ckpt"

def train(mnist):
    # 将处理输入数据的计算都放在名字为“input”的命名空间下
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = mnist_inference.inference(x, regularizer)

    global_step = tf.Variable(0, trainable=False)

    # 将处理欢动平均相关的计算都放在名为’moving_average‘的空间下
    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 将计算损失函数相关的计算都放在名为“loss_function”下
    with tf.name_scope("loss_function"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 将定义学习率，优化方法以及每一轮训练需要执行的操作都放在名为“train_step”的命名空间下
    with tf.name_scope('train_step'):
        learning_rate = tf.train.exponential_decay(
            LEARNINT_RATE_BASE,
            global_step,
            mnist.train.num_examples / BATCH_SIZE,
            LEARNINT_RATE_DECAY,
            staircase=True
        )
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
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
    # 将当期的计算图输入到TensorBoard日志文件中
    writer = tf.summary.FileWriter("C:/Wang_Work/Pycharm_workSpace/Demo1/chapter_9/log", graph=tf.get_default_graph())
    writer.close()

def main(argv=None):
    mnist = input_data.read_data_sets("C:/Wang_File/MNIST", one_hot=True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()

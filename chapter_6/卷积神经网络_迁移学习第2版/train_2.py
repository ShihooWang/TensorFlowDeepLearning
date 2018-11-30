"""

完成 Inception-v3 模型迁移学习的过程

"""

import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim
import datetime

# 加载 TensorFlow-Slim 定义好的Inception-v3模型
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

# 处理好的数据
INPUT_DIR = "data/processed_multi"

# 保存训练好后的模型
TRAIN_FILE_DIR = "data/save/model"

# Google已训练好的模型
CKPT_FILE = "data/inception_v3.ckpt"

# 训练参数
LEARNING_RATE = 0.0001
STEPS = 300
BATCH = 8
N_CLASS = 5

# 在新的问题中 只需要讯号好模型中最后一层全连接的产生
CHECHEPOTION_EXCLUE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'
# 需要训练的网络层参数名称，在 fine-tuning 的过程就是最后的全连接层 这里给出的是参数的前缀
TRAINABLE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'

# 获取所有需要从谷歌训练好的模型中加载的参数
def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECHEPOTION_EXCLUE_SCOPES.split(',')]
    variables_to_restore = []
    # 枚举 inception-v3 模型中所有的参数，然后判断是否要从加载列表中移除
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore

# 获取所有需要训练的变量列表
def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(",")]
    variables_to_train = []
    # 枚举所有需要训练的参数的前缀
    for scope in scopes :
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.append(variables)
    return variables_to_train

def main(_):

    training_images = []
    training_labels = []
    validation_images = []
    validation_labels = []
    testing_images = []
    testing_labels = []

    processed_data = []
    files = os.listdir(INPUT_DIR)  # 列表推导式
    for file in files:
        file_path = os.path.join(INPUT_DIR, file)
        proce_data = np.load(file_path)
        # 横向合并数组 不改变维度
        # a = [[1,2,3],[4,5,6]]
        # b = [[1,1,1],[2,2,2]]
        # 横向合并 d = np.hstack((a,b))
        # d = array([[1, 2, 3, 1, 1, 1],[4, 5, 6, 2, 2, 2]])
        processed_data = np.hstack((processed_data, proce_data))

    training_images = processed_data[0]
    training_labels = processed_data[1]
    validation_images = processed_data[2]
    validation_labels = processed_data[3]
    testing_images = processed_data[4]
    testing_labels = processed_data[5]
    n_training_image = len(training_images)
    print(" %d train_data, %d validation-data, %d test_data" %
          (n_training_image, len(validation_labels), len(testing_labels)))

    # 定义 InceptionV3 的输入，images 为输入图片，labels为每一张图片对应的标签
    images = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_images')
    labels = tf.placeholder(tf.int64, [None], name='labels')

    # 定义 InceptionV3 的结构模型
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, num_classes=N_CLASS)
    # 获取需要训练的变量
    trainable_variables = get_trainable_variables()
    # 定义交叉熵损失
    tf.losses.softmax_cross_entropy(tf.one_hot(labels, N_CLASS), logits, weights=1.0)
    # 定义训练过程
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss())
    # 计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 定义加载模型
    load_fn = slim.assign_from_checkpoint_fn(
        CKPT_FILE,
        get_tuned_variables(),
        ignore_missing_vars=True
    )
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # 加载模型
        load_fn(sess)
        start = 0
        end = BATCH
        for i in range(STEPS):
            # 训练过程 更新指定部分参数
            # 数据已经打乱过
            sess.run(train_step, feed_dict={
                images: training_images[start:end],
                labels: training_labels[start:end]
            })
            if i % 30 == 0 or i + 1 == STEPS:
                saver.save(sess, TRAIN_FILE_DIR, global_step=i)
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    images : validation_images,
                    labels : validation_labels
                })
                nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
                print('%s step %d : Validation accuracy = %.1f%%' % (nowTime, i, validation_accuracy * 100))
            start = end
            if start == n_training_image:
                start = 0
            end = start + BATCH
            if end > n_training_image:
                end = n_training_image

        # 在测试集上测试正确率
        test_accuracy = sess.run(evaluation_step, feed_dict={
            images: testing_images, labels: testing_labels})
        print('%s Final test accuracy = %.1f%%' % (nowTime, test_accuracy * 100))

if __name__ == '__main__':
    tf.app.run()
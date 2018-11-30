import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from chapter_5.最佳手写体识别 import mnist_inference
from chapter_5.最佳手写体识别 import mnist_train

# 每10秒加载一次新的模型，并在测试数据上测试最新模型的正确率
EVAL_INTERAL_SECS = 10

# 使用验证数据来验证
def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NONE], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NONE], name="y-input")
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}

        # 直接使用定义好的前向传播函数，注意 验证时不关注正则化损失
        y = mnist_inference.inference(x, None)
        # 使用rf.argmax函数来对位置的样例进行分类
        current_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(current_prediction, tf.float32))

        # variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variable_averages = tf.train.ExponentialMovingAverage(0.98)
        variables_to_store = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_store)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名获取到迭代次数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split("-")[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy =  %g ." % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERAL_SECS)

def main(argv = None):
    mnist = input_data.read_data_sets("C:/Wang_File/MNIST", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()

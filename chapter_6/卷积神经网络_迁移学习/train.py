import os.path
import tensorflow as tf
from chapter_6.卷积神经网络_迁移学习 import constant
from chapter_6.卷积神经网络_迁移学习 import sort_data
from tensorflow.python.platform import gfile
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main(_):
    # 读取所有图片
    image_lists = sort_data.create_image_list(constant.TEST_PERCENTAGE, constant.VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())

    with gfile.FastGFile(os.path.join(constant.MODEL_DIR, constant.MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())


    bottleneck_tensor, jepg_data_tensor = tf.import_graph_def(graph_def,
                                                                 return_elements=[constant.BOTTLENECK_TENSOR_NAME,
                                                                                  constant.JEPG_DATA_TENSOR_NAME])
    # 定义新的神经网络输入
    bottleneck_input = tf.placeholder(tf.float32, [None, constant.BOTTLENECK_TENSOR_SIZE],
                                       name="BottleneckInputPlaceholder")
    # 定义新的标准答案输入
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name="GroundTruthInput")

    # 定义一层新的全连接层来解决图片分类问题
    with tf.name_scope('final_training_ops'):
            # weights = tf.Variable(tf.truncated_normal(
            #     [constant.BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
            # biases = tf.Variable(tf.zeros([n_classes]))

        weights = tf.get_variable(
                "weights", [constant.BOTTLENECK_TENSOR_SIZE, n_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [n_classes], initializer=tf.constant_initializer(0.0))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)
    # 交叉熵 损失函数 反向传播
    # cross_etropy = tf.nn.softmax_cross_entropy_with_logits(logits, ground_truth_input)
    # cross_etropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(ground_truth_input, 1))
    cross_etropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_etropy_mean = tf.reduce_mean(cross_etropy)
    train_step = tf.train.GradientDescentOptimizer(constant.LEARNING_RATE).minimize(cross_etropy_mean)

    # 计算正确率
    with tf.name_scope('valuation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 训练过程
        for i in range(constant.STEPS):
            train_bottlenecks, train_ground_truth = sort_data.get_random_cached_bottlenecks(
                  sess, n_classes, image_lists, constant.BATCH, "training", jepg_data_tensor,bottleneck_tensor
              )
            sess.run(train_step, feed_dict={bottleneck_input: train_bottlenecks,
                                              ground_truth_input: train_ground_truth})

            if i % 100 == 0 or i+1 == constant.STEPS:
                validation_bottlenecks, validation_ground_truth = sort_data.get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, constant.BATCH, "validation", jepg_data_tensor, bottleneck_tensor
                )
                validation_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: validation_bottlenecks,
                                                     ground_truth_input: validation_ground_truth})
                print("After %d validation step(s), loss on validation average batch is %g ." % (i, validation_accuracy*100))
        # 最后在测试数据上跑一边
        test_bottlenecks, test_ground_truth = sort_data.get_test_bottlenecks(
            sess, image_lists, n_classes, jepg_data_tensor, bottleneck_tensor
        )
        test_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: test_bottlenecks,
                                                                   ground_truth_input: test_ground_truth})
        print("Final test accuracy is %g ." % (test_accuracy * 100))

        # 保存该模型
        saver.save(sess, constant.SESS_DIR)
if __name__ == '__main__':
    tf.app.run()
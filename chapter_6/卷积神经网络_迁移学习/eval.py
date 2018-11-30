"""
使用单个图片来验证上述 模型

"""
import os
from tensorflow.python.platform import gfile
from chapter_6.卷积神经网络_迁移学习 import sort_data
from chapter_6.卷积神经网络_迁移学习 import constant
import tensorflow as tf

def evaluate():
    image_path = input("请输入图片的路径：")
    if not os.path.exists(image_path):
        image_path = input("请输入正确的图片的路径：")

    with gfile.FastGFile(os.path.join(constant.MODEL_DIR, constant.MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    bottleneck_tensor, jepg_data_tensor = tf.import_graph_def(graph_def,
                            return_elements=[constant.BOTTLENECK_TENSOR_NAME, constant.JEPG_DATA_TENSOR_NAME])
    image_data = gfile.FastGFile(image_path, 'rb').read()

    x = tf.placeholder(tf.float32, [None, constant.BOTTLENECK_TENSOR_SIZE], name="x-input")
    weights = tf.get_variable(
        "weights", [constant.BOTTLENECK_TENSOR_SIZE, 5],
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable('biases', [5], initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(x, weights) + biases
    final_tensor = tf.nn.softmax(logits)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # ckpt = tf.train.get_checkpoint_state()
        # if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, constant.SESS_DIR)
            xs = sort_data.run_bottleneck_on_image(sess, image_data, jepg_data_tensor, bottleneck_tensor)
            xxs = []
            xxs.append(xs)
            value_r = sess.run(final_tensor, feed_dict={x: xxs})
            print("dandelion  蒲公英: %6f" % (value_r[0][0]))
            print("roses      玫瑰花: %6f" % (value_r[0][1]))
            print("daisy      雏菊  : %6f" % (value_r[0][2]))
            print("sunflowers 向日葵: %6f" % (value_r[0][3]))
            print("tulips     郁金香: %6f" % (value_r[0][4]))

def main(argv = None):

    evaluate()

if __name__ == '__main__':
    tf.app.run()



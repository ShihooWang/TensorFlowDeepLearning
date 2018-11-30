"""
将 MNIST 输入数据转化为 TFRecord 格式
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

mnist = input_data.read_data_sets("MNIST", dtype=tf.uint8, one_hot=True)
images = mnist.train.images
# 训练数据所对应的正确答案，可以作为一个属性保存在TFRecord中
labels = mnist.train.labels
# 训练数据的图像分辨率，可以作为Example中的一个属性
pixels = images.shape[1]
num_examples = mnist.train.num_examples

# 输出TFRecord文件地址 chapter_7/record/output.records
fileName = "record/output.records"
writer = tf.python_io.TFRecordWriter(fileName)
for index in range(num_examples):
    # 将图像矩阵转换成一个字符串
    image_raw = images[index].tostring()
    # 将一个样例转换为Example Protocol Buffer 并将所有的信息写进这个数据结构中
    example = tf.train.Example(
        features=tf.train.Features(
            feature={'pixels': _int64_feature(pixels),
                     'labels': _int64_feature(np.argmax(labels[index])),
                     'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
    if index % 1000 == 0:
        print(" 当前第 ：%d 步" % index)
writer.close()



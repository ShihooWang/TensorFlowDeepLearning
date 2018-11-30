import tensorflow as tf

# 创建一个reader来读取TFRecord文件中的样例
reader = tf.TFRecordReader()
# 创建一个队列来维护输入文件列表，
filename_queue = tf.train.string_input_producer(["record/output.records"])

#从文件中读取一个样例。也可以使用read_up_to一次性读取多个
_, serialized_example = reader.read(filename_queue)
# 解析读入的一个样例，如果要解析多个 使用parse_example函数
features = tf.parse_single_example(serialized_example,
                                   features={
                                       # 解析要和之前的写入格式一样 这里使用了 tf.FixedLenFeature 解析结果得到一个Tensor
                                       # 另一种方法 tf.VarLenFeature 解析结果为 SparseTensor 用于处理稀疏数据
                                        "image_raw": tf.FixedLenFeature([], tf.string),
                                        "pixels": tf.FixedLenFeature([], tf.int64),
                                        "labels": tf.FixedLenFeature([], tf.int64)
                                    })
# tf.decode_raw 可将字符串解析成图像对应的像素数组
images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['labels'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()
# 启动多线程处理输入数据
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 每次运行可以读取TFRecord 文件中的一个样例，
for i in range(40000):
    image, label, pixel = sess.run([images, labels, pixels])
    if i % 1000 == 1:
        print(sess.run(labels)) # 打印的是当前手写体的类型 0-9

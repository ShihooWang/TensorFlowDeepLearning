"""
 用来将原始的图像数据整理成模型需要的输入数据

 本次使用 单线程 多文件分开保存
 每个花的种类保存一个npy格式文件
"""
import glob
import datetime
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# 原始数据目录下有5组数据
INPUT_DATA = "data/flower_photos"

# 测试数据和验证数据的比例
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

# 读取数据并将数据分割成训练数据、验证数据、测试数据
def create_image_list(sess,file_list, dir_name, current_label, testing_percentage, validation_percentage):

    # 初始化各个数据集
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []

    count = 0
    for file_name in file_list:
        # 读取并解析图片，将图片转换为299x299以便inception-v3模型来处理
        image_raw_data = gfile.FastGFile(file_name, 'rb').read()
        image = tf.image.decode_jpeg(image_raw_data)
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, [299, 299])
        image_value = sess.run(image)

        # 随机划分数据集
        chance = np.random.randint(100)
        if chance < validation_percentage:
            validation_images.append(image_value)
            validation_labels.append(current_label)
        elif chance < (testing_percentage + validation_percentage):
            testing_images.append(image_value)
            testing_labels.append(current_label)
        # elif chance < 50:
        #     training_images.append(image_value)
        #     training_labels.append(current_label)
        else:
            training_images.append(image_value)
            training_labels.append(current_label)
        count += 1
        if count % 50 == 0 :
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
            print("%s 执行了 %d 次" % (nowtime, count))

    # 将训练数据随机打乱
    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)

    processed_data = np.asarray([training_images, training_labels,
                                 validation_images, validation_labels,
                                 testing_images, testing_labels])
    # 通过numpy格式保存处理后的数据
    # 输出文件地址 整理后的图片数据通过numpy格式保存
    file_dir = "data/processed_multi"
    # 先创建文件夹
    if not os.path.exists(file_dir): os.makedirs(file_dir)
    output_file = "data/processed_multi/flower_data_" + dir_name + ".npy"
    # open 打开一个文件，而不是文件夹（目录）
    np.save(output_file, processed_data)

def main():
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]  # 列表推导式
    is_root_dir = True

    current_label = 0

    # 读取所有的子目录
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取一个子目录中的所有文件
        # extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']  # 后缀名
        extensions = ['jpg', 'jpeg']  # 后缀名
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue  # 判断数组是否为空
        print(" 准备读取 %s 文件夹" % dir_name)

        with tf.Session() as sess:
            create_image_list(sess, file_list, dir_name, current_label, TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
        # 当前分类标签 +1 的操作 定义在大循环外
        current_label += 1

if __name__ == '__main__':
    main()









